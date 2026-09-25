from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TokenizersBackend,
)
from omegaconf import DictConfig, OmegaConf
import hydra
from datasets import load_dataset
from functools import partial
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass


def transform_batch(
    batch,
    context_tokenizer: TokenizersBackend,
    state_tokenizer: TokenizersBackend,
):
    messages = [json.loads(message) for message in batch["messages"]]

    tokenized_context = context_tokenizer.apply_chat_template(
        messages,
        truncation=True,
    )

    tokenized_state = state_tokenizer.apply_chat_template(
        messages,
        truncation=True,
        return_assistant_tokens_mask=True,
    )

    return {
        "context_input_ids": tokenized_context["input_ids"],
        "context_attention_mask": tokenized_context["attention_mask"],
        "state_input_ids": tokenized_state["input_ids"],
        "state_attention_mask": tokenized_state["attention_mask"],
        "state_assistant_mask": tokenized_state["assistant_masks"],
        "resolved": batch["resolved"],
    }


class MyDataCollator:
    def __init__(
        self,
        context_tokenizer: TokenizersBackend,
        state_tokenizer: TokenizersBackend,
    ):
        self.context_tokenizer = context_tokenizer
        self.state_tokenizer = state_tokenizer

    def __call__(self, features):
        padded_context = self.context_tokenizer.pad(
            {
                "input_ids": [feature["context_input_ids"] for feature in features],
                "attention_mask": [feature["context_attention_mask"] for feature in features],
            },
            padding_side="left",
            return_tensors="pt",
        )

        padded_state = self.state_tokenizer.pad(
            {
                "input_ids": [feature["state_input_ids"] for feature in features],
                "attention_mask": [feature["state_attention_mask"] for feature in features],
            },
            padding_side="right",
            return_tensors="pt",
        )

        state_assistant_mask = pad_sequence(
            [torch.tensor(feature["state_assistant_mask"]) for feature in features],
            padding_value=0,
            padding_side="right",
            batch_first=True,
        )

        resolved = [feature["resolved"] for feature in features]

        return {
            "context_input_ids": padded_context["input_ids"],
            "context_attention_mask": padded_context["attention_mask"],
            "state_input_ids": padded_state["input_ids"],
            "state_attention_mask": padded_state["attention_mask"],
            "state_assistant_mask": state_assistant_mask,
            "resolved": resolved,
        }


@dataclass
class MyTrainingArguments(TrainingArguments):
    chunked_kl_div_chunk_size: int = 1024


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.backbone = unwrapped_model.get_decoder()
        self.head = unwrapped_model.get_output_embeddings()

        def forward(*args, **kwargs):
            return self.backbone(*args, **kwargs).last_hidden_state
        
        unwrapped_model.forward = forward

    def _get_num_items_in_batch(self, batch_samples, device):
        return sum(batch["state_assistant_mask"][:, 1:].sum() for batch in batch_samples).to(device)

    def compute_loss(
        self,
        model,
        inputs,
        num_items_in_batch,
        return_outputs=False,
    ):
        with torch.no_grad():
            model.eval()
            teacher_last_hidden_state = model(
                input_ids=torch.concat([inputs["context_input_ids"], inputs["state_input_ids"]], dim=1),
                attention_mask=torch.concat([inputs["context_attention_mask"], inputs["state_attention_mask"]], dim=1),
            )
            model.train()

        student_last_hidden_state = model(
            input_ids=inputs["state_input_ids"],
            attention_mask=inputs["state_attention_mask"],
        )

        assistant_mask = inputs["state_assistant_mask"]

        shifted_teacher_last_hidden_state = teacher_last_hidden_state[:, inputs["context_input_ids"].shape[1]:-1]
        shifted_student_last_hidden_state = student_last_hidden_state[:, :-1]
        shifted_assistant_mask = assistant_mask[:, 1:]

        def compute_loss(
            teacher_last_hidden_state,
            student_last_hidden_state,
            assistant_mask,
        ):
            with torch.no_grad():
                self.head.eval()
                teacher_logits = self.head(teacher_last_hidden_state)
                self.head.train()

            student_logits = self.head(student_last_hidden_state)

            kl_divergence = torch.nn.functional.kl_div(
                torch.log_softmax(student_logits, dim=-1),
                torch.softmax(teacher_logits, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            return (kl_divergence * assistant_mask).sum()
        
        chunk_size = self.args.chunked_kl_div_chunk_size
        loss = 0
        for i in range(0, shifted_teacher_last_hidden_state.shape[1], chunk_size):
            loss += torch.utils.checkpoint.checkpoint(
                compute_loss,
                shifted_teacher_last_hidden_state[:, i:i+chunk_size],
                shifted_student_last_hidden_state[:, i:i+chunk_size],
                shifted_assistant_mask[:, i:i+chunk_size],
                use_reentrant=False,
            )
        return loss / num_items_in_batch


@hydra.main(version_base="1.2", config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    model = AutoModelForCausalLM.from_pretrained(
        **OmegaConf.to_container(cfg.model, resolve=True)
    )

    context_tokenizer = AutoTokenizer.from_pretrained(
        **OmegaConf.to_container(cfg.tokenizer, resolve=True),
        truncation_side="left",
    )
    if context_tokenizer.pad_token_id is None:
        context_tokenizer.pad_token = context_tokenizer.eos_token

    state_tokenizer = AutoTokenizer.from_pretrained(
        **OmegaConf.to_container(cfg.tokenizer, resolve=True),
        truncation_side="right",
    )
    if state_tokenizer.pad_token_id is None:
        state_tokenizer.pad_token = state_tokenizer.eos_token

    dataset = load_dataset(**OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset = dataset.map(
        partial(
            transform_batch,
            context_tokenizer=context_tokenizer,
            state_tokenizer=state_tokenizer,
        ),
        **OmegaConf.to_container(cfg.dataset_map, resolve=True),
        batched=True,
    )

    data_collator = MyDataCollator(
        context_tokenizer=context_tokenizer,
        state_tokenizer=state_tokenizer,
    )

    args = MyTrainingArguments(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        remove_unused_columns=False,
    )
    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
