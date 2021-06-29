from dataclasses import dataclass, field
from io import StringIO
from collections import Counter

import re
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import Color

import spacy
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch
import torch.nn.functional as F
import streamlit as st
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server

######################################## INITIAL CONFIG ##########################################
st.set_page_config(
    page_title="MEDTEXT NLP",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded",
)

size = 15
params = {
    "legend.fontsize": "large",
    "figure.figsize": (20, 10),
    "axes.labelsize": 25,
    "axes.titlesize": 25,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlepad": 25,
}
plt.rcParams["font.sans-serif"] = ["Avenir", "sans-serif"]
plt.rcParams.update(params)


st.title("MEDTEXT NLP")
col1, col2 = st.beta_columns((1, 3))

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


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.load_state_dict(
        torch.load("trained_models/medtext-final.pt", map_location=torch.device("cpu"))
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return model, tokenizer


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_ner_tagger(tagger):
    return spacy.load(tagger)


@st.cache(allow_output_mutation=True)
def load_comment(text):
    return text


def b_or_w_font(hex_value):
    color = Color(f"#{hex_value}")

    if color.get_luminance() > 0.45:
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


def justify_text(text):
    return f'<div style="text-align: justify"> {text} </div>'


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


def display_analysis(text, doc, col_dict):
    with st.beta_expander("Mostrar más datos", expanded=True):
        st.write("### Algunas gráficas y estadísticas de los comentarios...")

        num_tokens = len(text.split(" "))
        num_chars = len(text)

        st.markdown("# Tamaño de nuestro comentario:")

        st.markdown(f"\t### {num_tokens} tokens")
        st.markdown(f"\t### {num_chars} caracteres")

        st.markdown("---")

        text_data = [(ent.text, ent.label_) for ent in doc.ents]
        counter = Counter([t[1] for t in text_data])

        if len(counter.items()) > 0:
            st.markdown("# Se encontraron las siguientes etiquetas...")
            for k, v in counter.items():
                st.markdown(f"### {v} {k}")

            if len(counter.items()) > 1:
                df = pd.DataFrame(counter.items(), columns=["Tag", "Count"])

                fig = plt.figure(figsize=(10, 5))
                plt.barh(df.Tag, df.Count, color=[col_dict[tag] for tag in df.Tag])
                st.pyplot(fig)
        else:
            st.markdown("## El modelo no encontró ninguna etiqueta.")
            st.markdown(
                "#### Puedes probar a cambiar el modelo arriba, ofrecerá otros resultados."
            )


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
            st.markdown(justify_text(c), unsafe_allow_html=True)
            st.markdown("---")

        if st.button("Borrar comentarios"):
            state.clear()


# EVALUATION COLUMN
def comment_evaluation(state):
    st.header("EVALUAR COMENTARIOS")

    selection = st.selectbox(
        "Elige el modelo a utilizar:",
        ["en_core_med7_trf", "en_ner_bionlp13cg_md", "en_ner_bc5cdr_md"],
        1,
    )

    ner_tagger = load_ner_tagger(selection)

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

    texts = None
    if source_comment_choice == "Generados":
        if state.generated_comments is not None:
            texts = state.generated_comments
        else:
            st.warning("Debes generar comentarios primero!")

    elif source_comment_choice == "Desde archivo":
        st.warning(
            "Sube un archivo de texto con tus comentarios aquí. Debe ser un .txt o .dat. Se espera que haya un comentario por línea."
        )
        uploaded_file = st.file_uploader("", type=["txt", "dat"])
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            texts = stringio.readlines()

    elif source_comment_choice == "Escribir":
        st.subheader(
            "Puedes escribir tus comentarios aquí. Escribe un solo comentario por línea."
        )
        texts = st.text_area("Escribe aquí tus comentarios.", height=200).split("\n")

    if texts is None:
        pass

    else:

        if len(texts) == 1:
            state.text_idx = 0

        else:
            state.text_idx = st.selectbox(
                "Elige el comentario a analizar", options=list(range(len(texts)))
            )

        state.text = load_comment(texts[state.text_idx])

        doc = ner_tagger(state.text)

        styled_html = spacy.displacy.render(
            doc, style="ent", options=html_format_options
        )

        styled_html = check_colors_html(styled_html)

        st.markdown(styled_html, unsafe_allow_html=True)

        st.markdown("---")

        display_analysis(state.text, doc, col_dict)


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
