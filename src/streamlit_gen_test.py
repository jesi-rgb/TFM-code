from dataclasses import dataclass, field
import re
from io import StringIO
import streamlit as st
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import torch.nn.functional as F
import spacy

from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from streamlit.hashing import _CodeHasher
from colour import Color


######################################## INITIAL CONFIG ##########################################
st.set_page_config(
    page_title="MEDTEXT NLP",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("MEDTEXT NLP")
col1, col2 = st.beta_columns((1, 1))

######################################## FUNCTIONS AND CLASSES ##########################################
class _SessionState:
    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(
                self._state["data"], None
            ):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


@st.cache
@dataclass(frozen=True)
class GeneratedComment:
    comment_list: list[str] = field(default_factory=list)

    def __getitem__(self, item):
        return self.comment_list[item]


def normalize(n, n_min=0, n_max=1):
    return (n - n_min) / (n_max - n_min)


@st.cache(allow_output_mutation=True)
def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.load_state_dict(
        torch.load("trained_models/medtext-final.pt", map_location=torch.device("cpu"))
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return model, tokenizer


@st.cache(allow_output_mutation=True)
def load_med7():
    return spacy.load("en_core_med7_trf")


@st.cache(allow_output_mutation=True)
def load_en_ner_bionlp13cg_md():
    return spacy.load("en_ner_bionlp13cg_md")


def b_or_w_font(hex_value):
    color = Color(f"#{hex_value}")

    print(color.get_luminance())
    if color.get_luminance() > 0.40:
        value = "#000000"
    else:
        value = "#ffffff"
    return value


def check_colors_html(styled_html):
    color_list = re.findall(r"(?<=background: #)\w+", styled_html)
    text_color_list = [b_or_w_font(c) for c in color_list]

    for c, t in zip(color_list, text_color_list):
        styled_html = re.sub(f"#{c};", f"#{c}; color: {t};", styled_html)

    return styled_html


def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=100,
    top_p=0.8,
    temperature=1.0,
    progress_bar=None,
    progress_text=None,
):

    model.eval()

    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in range(entry_count):
            current_stop = entry_idx

            entry_finished = False

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            # Using top-p (nucleus sampling): https://github.com/huggingface/transformers/blob/master/examples/run_generation.py

            progress_text.write(f"{entry_idx + 1} / {entry_count} comentarios...")
            for i in range(entry_length):

                current_word = normalize(
                    current_stop + i / entry_length,
                    n_max=entry_count,
                )

                progress_bar.progress(current_word)

                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|EOS|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)

                    generated_list.append(output_text)
                    break

            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<|EOS|>"
                generated_list.append(output_text)

    return generated_list


######################################## PAGE DEFINITION ##########################################

# GENERATION COLUMN


def comment_generation(state):
    st.header("GENERAR COMENTARIOS")
    n_comments = st.number_input("Nº de comentarios a generar", 1, 5, 2, 1)

    if st.button("Generar comentarios"):

        with st.spinner("Cargando el modelo en memoria"):

            model, tokenizer = load_model()

        st.success("Modelo cargado!")

        coms = st.empty()
        coms.write("Generando comentarios... puede tomar un rato")
        progress = st.progress(0)
        progress_text = st.empty()

        generated_comments = GeneratedComment(
            generate(
                model.to("cpu"),
                tokenizer,
                "<|BOS|>",
                entry_count=n_comments,
                progress_bar=progress,
                progress_text=progress_text,
            )
        )

        progress.empty()
        progress_text.empty()
        coms.empty()

        generated_comments = [
            l.removeprefix("<|BOS|>").rstrip("<|EOS|>\n") for l in generated_comments
        ]

        state.generated_comments = generated_comments

    if state.generated_comments is not None:
        st.markdown("---")
        for c in state.generated_comments:
            st.markdown(c)
            st.markdown("---")

        if st.button("Borrar comentarios"):
            state.clear()


# EVALUATION COLUMN
def comment_evaluation(state):
    st.header("EVALUAR COMENTARIOS")

    selection = st.selectbox(
        "Elige el modelo a utilizar:", ["med7", "en_ner_bionlp13cg_md"], 1
    )

    if selection == "med7":
        ner_tagger = load_med7()
    else:
        ner_tagger = load_en_ner_bionlp13cg_md()

    # configure the entities parser colours
    col_dict = {}
    seven_colours = [
        "#e6194B",
        "#3cb44b",
        "#ffe119",
        "#ffd8b1",
        "#f58231",
        "#f032e6",
        "#42d4f4",
        "#ff0000",
        "#ff8700",
        "#ffd300",
        "#deff0a",
        "#a1ff0a",
        "#0aff99",
        "#0aefff",
        "#147df5",
        "#580aff",
        "#be0aff",
    ]

    for label, colour in zip(ner_tagger.pipe_labels["ner"], seven_colours):
        col_dict[label] = colour

    html_format_options = {"ents": ner_tagger.pipe_labels["ner"], "colors": col_dict}

    source_comment_choice = st.radio(
        "Elige de dónde cargar los comentarios",
        ["Generados", "Desde archivo", "Escribir"],
        index=0,
    )

    text = None
    if source_comment_choice == "Generados":
        if state.generated_comments is not None:
            text = state.generated_comments
        else:
            st.warning("Debes generar comentarios primero!")

    elif source_comment_choice == "Desde archivo":
        st.warning(
            "Sube un archivo de texto con tus comentarios aquí. Debe ser un .txt o .dat. Se espera que haya un comentario por línea."
        )
        uploaded_file = st.file_uploader("", type=["txt", "dat"])
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.readlines()

    elif source_comment_choice == "Escribir":
        st.subheader(
            "Puedes escribir tus comentarios aquí. Escribe un solo comentario por línea."
        )
        text = st.text_area("Escribe aquí tus comentarios.", height=200).split("\n")

    if st.button("Analizar comentarios"):
        if text is None:
            pass

        elif len(text) == 1:
            doc = ner_tagger(text[0])

            styled_html = spacy.displacy.render(
                doc, style="ent", options=html_format_options
            )

            styled_html = check_colors_html(styled_html)

            st.markdown(styled_html, unsafe_allow_html=True)

        else:
            docs = list(ner_tagger.pipe(text))
            doc_renders = [
                spacy.displacy.render(doc, style="ent", options=html_format_options)
                for doc in docs
            ]

            styled_renders = [check_colors_html(render) for render in doc_renders]

            for doc in styled_renders:
                st.markdown(doc, unsafe_allow_html=True)
                st.markdown("---")


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


def main():
    state = _get_state()
    pages = {
        "Generation": comment_generation,
        "Evaluation": comment_evaluation,
    }

    # Display the selected page with the session state
    with col1:
        pages["Generation"](state)

    with col2:
        pages["Evaluation"](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


if __name__ == "__main__":
    main()
