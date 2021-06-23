#!/usr/bin/python3
import torch
import os
from torch._C import device
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch.nn.functional as F
from tqdm import tqdm, trange
import gc


class MedTextDataset(Dataset):
    def __init__(self, *path_files, gpt2_type="gpt2", max_length=768):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.comments = []

        for path in path_files:
            print(f"Reading file {path} ...")
            with open(path) as file:
                curr_file = file.readlines()

                self.comments.extend(
                    [self.comment_to_tensor(c, max_length) for c in curr_file]
                )

        self.comments_count = len(self.comments)

    def read_file(self, path):
        with open(path) as file:
            return file.readlines()

    def comment_to_tensor(self, comment, max_length):
        return torch.tensor(self.tokenizer.encode(comment[:max_length]))

    def __len__(self):
        return self.comments_count

    def __getitem__(self, item):
        return self.comments[item]


def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


def train(
    dataset,
    model,
    batch_size=16,
    epochs=4,
    lr=2e-5,
    warmup_steps=5000,
    device="cuda",
    output_dir=".",
    output_prefix="medtex",
    save_model_on_epoch=False,
):

    torch.cuda.empty_cache()

    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Epoch {epoch+1}\n-------------------------------")
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )

        # save the last model anyways
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"{output_prefix}-final.pt"),
        )
    return model


def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=100,
    top_p=0.8,
    temperature=1.0,
):

    model.eval()

    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            # Using top-p (nucleus sampling): https://github.com/huggingface/transformers/blob/master/examples/run_generation.py

            for i in range(entry_length):
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


if __name__ == "__main__":
    # clean the machine's RAM before processing, to save on as much space as possible
    gc.collect()
    torch.cuda.empty_cache()

    # declaring variables, model names and paths
    spaces = "-" * 50
    DATA_FOLDER = "tagged_files"
    gpt2_type = "gpt2"

    print("Starting GPT2 fine-tuning with med-text data.")

    files = os.listdir(DATA_FOLDER)

    print(f"Loading data from {DATA_FOLDER}")
    train_p_dataset = MedTextDataset(*[os.path.join(DATA_FOLDER, f) for f in files])
    print(f"Processed {len(train_p_dataset)} sentences.")

    print(f"\n{spaces}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Starting training...")
    model = train(
        train_p_dataset,
        GPT2LMHeadModel.from_pretrained(gpt2_type),
        batch_size=64,
        epochs=3,
        lr=2e-5,
        warmup_steps=200,
        device=device,
        output_dir="trained_models",
        output_prefix="medtext",
        save_model_on_epoch=False,
    )

    print("Training finished.")

    print()
    print("*" * 100)
    print()

    n_entries = 10
    print("Starting generation of comments...")
    print(f"Generating {n_entries} sentences...")
    generated_comments = generate(
        model.to("cpu"),
        GPT2Tokenizer.from_pretrained(gpt2_type),
        "<|BOS|>",
        entry_count=n_entries,
    )

    with open("results/generated_comments.txt", "w") as file:
        file.write("\n".join(generated_comments))

    print(f"\n\n{spaces}\n\n".join(generated_comments))
    print("Execution finished")
