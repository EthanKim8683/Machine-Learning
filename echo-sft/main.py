import json
from dataclasses import dataclass
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
USER_TOKEN = "user"
ASSISTANT_TOKEN = "assistant"


def tokenize_dataset(dataset, tokenizer):
    [im_start_id] = tokenizer.encode(IM_START_TOKEN)
    [im_end_id] = tokenizer.encode(IM_END_TOKEN)
    [user_id] = tokenizer.encode(USER_TOKEN)
    [assistant_id] = tokenizer.encode(ASSISTANT_TOKEN)

    def tokenize_batch(batch):
        batch_messages = [json.loads(messages) for messages in batch["messages"]]
        batch_input_ids = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            truncation=True,
            return_dict=False,
        )
        batch_action_labels = []
        batch_observation_labels = []
        for input_ids, resolved in zip(batch_input_ids, batch["resolved"]):
            action_labels = []
            observation_labels = []
            turn = ""
            for i, input_id in enumerate(input_ids):
                if turn == "user":
                    action_labels.append(-100)
                    observation_labels.append(input_id)
                elif turn == "assistant":
                    action_labels.append(input_id if resolved else -100)
                    observation_labels.append(-100)
                else:
                    action_labels.append(-100)
                    observation_labels.append(-100)

                if i - 1 >= 0 and input_ids[i - 1] == im_start_id:
                    if input_id == user_id:
                        turn = "user"
                    elif input_id == assistant_id:
                        turn = "assistant"
                    else:
                        turn = ""

                if input_id == im_end_id:
                    turn = ""

            batch_action_labels.append(action_labels)
            batch_observation_labels.append(observation_labels)

        return {
            "input_ids": batch_input_ids,
            "action_labels": batch_action_labels,
            "observation_labels": batch_observation_labels,
        }

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )


class ECHODataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        raw_batch_action_labels = []
        raw_batch_observation_labels = []
        for feature in features:
            raw_batch_action_labels.append(feature.pop("action_labels"))
            raw_batch_observation_labels.append(feature.pop("observation_labels"))

        batch = super().__call__(features, return_tensors)

        batch_action_labels = []
        batch_observation_labels = []
        for (
            input_ids,
            raw_action_labels,
            raw_observation_labels,
        ) in zip(
            batch["input_ids"],
            raw_batch_action_labels,
            raw_batch_observation_labels,
        ):
            padding = [self.label_pad_token_id] * (
                len(input_ids) - len(raw_action_labels)
            )
            if self.tokenizer.padding_side == "right":
                action_labels = raw_action_labels + padding
                observation_labels = raw_observation_labels + padding
            else:
                action_labels = padding + raw_action_labels
                observation_labels = padding + raw_observation_labels

            batch_action_labels.append(action_labels)
            batch_observation_labels.append(observation_labels)

        if return_tensors == "pt":
            batch_action_labels = torch.tensor(batch_action_labels)
            batch_observation_labels = torch.tensor(batch_observation_labels)
        elif return_tensors == "np":
            batch_action_labels = np.array(batch_action_labels)
            batch_observation_labels = np.array(batch_observation_labels)

        batch["action_labels"] = batch_action_labels
        batch["observation_labels"] = batch_observation_labels
        return batch


@dataclass
class ECHOSFTConfig(SFTConfig):
    observation_loss_weight: float = 0.1


class ECHOSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoother = LabelSmoother(
            epsilon=self.args.label_smoothing_factor,
            ignore_index=self.data_collator.label_pad_token_id,
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        action_labels = inputs.pop("action_labels")
        observation_labels = inputs.pop("observation_labels")

        outputs = model(**inputs)

        action_loss = self.label_smoother(
            outputs,
            action_labels,
            shift_labels=True,
            num_items_in_batch=num_items_in_batch,
        )
        observation_loss = self.label_smoother(
            outputs,
            observation_labels,
            shift_labels=True,
            num_items_in_batch=num_items_in_batch,
        )
        loss = action_loss + self.args.observation_loss_weight * observation_loss

        return (loss, outputs) if return_outputs else loss


@hydra.main(version_base="1.2", config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    model = AutoModelForCausalLM.from_pretrained(
        **OmegaConf.to_container(cfg.model, resolve=True)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        **OmegaConf.to_container(cfg.tokenizer, resolve=True)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(**OmegaConf.to_container(cfg.dataset))
    train_dataset = tokenize_dataset(train_dataset, tokenizer=tokenizer)

    data_collator = ECHODataCollator(
        **OmegaConf.to_container(cfg.data_collator, resolve=True),
        tokenizer=tokenizer,
    )

    args = ECHOSFTConfig(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = ECHOSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
