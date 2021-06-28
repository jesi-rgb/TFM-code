import streamlit as st
import torch
from torch._C import device
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import torch.nn.functional as F
from tqdm import tqdm, trange


st.set_page_config(
    page_title="MEDTEXT NLP",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MEDTEXT NLP")
col1, col2 = st.beta_columns((2, 2))


######################################## CODE ##########################################


def clamp(n, smallest=0, largest=1):
    return max(smallest, min(n, largest))


def normalize(n, n_min=0, n_max=1):
    return (n - n_min) / (n_max - n_min)


# TODO: cachear el modelo y tokenizador para evitar cargarlos mucho.
@st.cache(allow_output_mutation=True)
def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.load_state_dict(
        torch.load("trained_models/medtext-final.pt", map_location=torch.device("cpu"))
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return model, tokenizer


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


######################################## STREAMLIT APP ##########################################

# GENERATION COLUMN
with col1:
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

        generated_comments = generate(
            model.to("cpu"),
            tokenizer,
            "<|BOS|>",
            entry_count=n_comments,
            progress_bar=progress,
            progress_text=progress_text,
        )

        progress.empty()
        progress_text.empty()
        coms.empty()

        st.write(
            [l.removeprefix("<|BOS|>").rstrip("<|EOS|>\n") for l in generated_comments]
        )


# EVALUATION COLUMN
with col2:
    st.header("EVALUAR COMENTARIOS")
