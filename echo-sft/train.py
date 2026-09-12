import json
from dataclasses import dataclass, field
from typing import Any
import torch
import torch.nn.functional as F
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import load_dataset


IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
USER_TOKEN = "user"
ASSISTANT_TOKEN = "assistant"


def tokenize_dataset(dataset, tokenizer):
    (
        im_start_id,
        im_end_id,
        user_id,
        assistant_id,
    ) = tokenizer.convert_tokens_to_ids(
        [
            IM_START_TOKEN,
            IM_END_TOKEN,
            USER_TOKEN,
            ASSISTANT_TOKEN,
        ],
    )

    def tokenize_batch(batch):
        messages_batch = [json.loads(messages) for messages in batch["messages"]]
        input_ids_batch = tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            truncation=True,
            return_dict=False,
        )

        labels_batch = []
        action_mask_batch = []
        observation_mask_batch = []
        for input_ids, resolved in zip(input_ids_batch, batch["resolved"]):
            labels = []
            action_mask = []
            observation_mask = []
            role_id = -1
            for i, input_id in enumerate(input_ids):
                is_action = role_id == assistant_id and resolved
                is_observation = role_id == user_id

                labels.append(input_id if is_action or is_observation else -100)
                action_mask.append(1 if is_action else 0)
                observation_mask.append(1 if is_observation else 0)

                if i - 1 >= 0 and input_ids[i - 1] == im_start_id:
                    role_id = input_id
                if input_id == im_end_id:
                    role_id = -1

            labels_batch.append(labels)
            action_mask_batch.append(action_mask)
            observation_mask_batch.append(observation_mask)

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "action_mask": action_mask_batch,
            "observation_mask": observation_mask_batch,
        }

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )


class MyDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        action_mask_batch = []
        observation_mask_batch = []
        for feature in features:
            action_mask_batch.append(feature.pop("action_mask"))
            observation_mask_batch.append(feature.pop("observation_mask"))

        batch = super().__call__(features, return_tensors)

        def pad_mask_batch(mask_batch):
            padded_mask_batch = []
            for labels, mask in zip(batch["labels"], mask_batch):
                padding = [0] * (len(labels) - len(mask))
                if self.tokenizer.padding_side == "right":
                    padded_mask_batch.append(mask + padding)
                else:
                    padded_mask_batch.append(padding + mask)

            if return_tensors == "pt":
                return torch.tensor(padded_mask_batch)
            elif return_tensors == "np":
                return np.array(padded_mask_batch)
            return padded_mask_batch

        batch["action_mask"] = pad_mask_batch(action_mask_batch)
        batch["observation_mask"] = pad_mask_batch(observation_mask_batch)
        return batch


@dataclass
class MyTrainingArguments(TrainingArguments):
    observation_loss_weight: float = 0.1
    chunked_cross_entropy: bool = False
    chunked_cross_entropy_kwargs: dict[str, Any] = field(default_factory=dict)


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.micro_step = 0
        self.patch_model(self.model)

    def patch_model(self, model):
        if self.args.chunked_cross_entropy is False:
            return

        model = self.accelerator.unwrap_model(model)
        decoder = model.get_decoder()
        output_embeddings = model.get_output_embeddings()

        base_forward = model.forward

        def forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
            if labels is None:
                return base_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )

            kwargs["use_cache"] = False
            hidden_state = decoder(
                input_ids,
                attention_mask,
                **kwargs,
            ).last_hidden_state

            hidden_state = hidden_state[:, :-1, :]
            labels = labels[:, 1:]

            batch_size, seq_len, _ = hidden_state.shape
            chunk_size = self.args.chunked_cross_entropy_kwargs.get("chunk_size", 256)

            def compute_per_token_loss_chunk(hidden_state_chunk, labels_chunk):
                logits_chunk = output_embeddings(hidden_state_chunk)
                return F.cross_entropy(
                    logits_chunk.view(-1, logits_chunk.size(-1)),
                    labels_chunk.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(batch_size, -1)

            per_token_loss_chunks = []
            for start in range(0, seq_len, chunk_size):
                end = start + chunk_size

                hidden_state_chunk = hidden_state[:, start:end, :]
                labels_chunk = labels[:, start:end]

                if self.args.gradient_checkpointing:
                    gradient_checkpointing_kwargs = (
                        self.args.gradient_checkpointing_kwargs or {}
                    )
                    use_reentrant = gradient_checkpointing_kwargs.get(
                        "use_reentrant",
                        False,
                    )

                    per_token_loss_chunk = torch.utils.checkpoint.checkpoint(
                        compute_per_token_loss_chunk,
                        hidden_state_chunk,
                        labels_chunk,
                        use_reentrant=use_reentrant,
                    )
                else:
                    per_token_loss_chunk = compute_per_token_loss_chunk(
                        hidden_state_chunk,
                        labels_chunk,
                    )
                per_token_loss_chunks.append(per_token_loss_chunk)

            per_token_loss = torch.cat(per_token_loss_chunks, dim=-1)

            return CausalLMOutputWithPast(loss=per_token_loss, logits=None)

        model.forward = forward

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        self.micro_step += 1

        action_mask = inputs.pop("action_mask")
        observation_mask = inputs.pop("observation_mask")

        outputs = model(**inputs)
        per_token_loss = outputs.loss

        logs = {"micro_step": self.micro_step}

        def compute_and_log_masked_loss(loss_name, mask):
            mask = mask[:, 1:]
            masked_loss = (per_token_loss * mask).sum()

            mask_sum = mask.sum()
            if mask_sum != 0:
                logs[loss_name] = (masked_loss / mask_sum).detach().item()

            return masked_loss

        action_loss = compute_and_log_masked_loss(
            "action_loss",
            action_mask,
        )
        observation_loss = compute_and_log_masked_loss(
            "observation_loss",
            observation_mask,
        )

        self.log(logs)

        loss = action_loss + self.args.observation_loss_weight * observation_loss
        loss = loss / (num_items_in_batch or inputs["labels"].size(-1))
        return (loss, outputs) if return_outputs else loss


@hydra.main(version_base="1.2", config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    model = AutoModelForCausalLM.from_pretrained(
        **OmegaConf.to_container(cfg.model, resolve=True)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        **OmegaConf.to_container(cfg.tokenizer, resolve=True)
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(**OmegaConf.to_container(cfg.dataset))
    train_dataset = tokenize_dataset(train_dataset, tokenizer=tokenizer)

    data_collator = MyDataCollator(
        **OmegaConf.to_container(cfg.data_collator, resolve=True),
        tokenizer=tokenizer,
    )

    args = MyTrainingArguments(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        remove_unused_columns=False,
    )
    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
