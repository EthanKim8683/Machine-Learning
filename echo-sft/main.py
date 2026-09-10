import json
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    model = AutoModelForCausalLM.from_pretrained(**OmegaConf.to_container(cfg.model))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    [im_start_id] = tokenizer.encode("<|im_start|>")
    [im_end_id] = tokenizer.encode("<|im_end|>")
    [user_id] = tokenizer.encode("user")
    [assistant_id] = tokenizer.encode("assistant")

    def get_action_and_observation_labels(input_ids):
        action_labels = []
        observation_labels = []
        turn = ""
        for i, input_id in enumerate(input_ids):
            if turn == "user":
                action_labels.append(-100)
                observation_labels.append(input_id)
            elif turn == "assistant":
                action_labels.append(input_id)
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

        return action_labels, observation_labels

    def tokenize_batch(batch):
        batch_messages = [json.loads(message) for message in batch["messages"]]
        raw_batch_input_ids = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            return_dict=False,
        )

        batch_input_ids = []
        batch_labels = []
        for input_ids, resolved in zip(raw_batch_input_ids, batch["resolved"]):
            (
                action_labels,
                observation_labels,
            ) = get_action_and_observation_labels(input_ids)

            if resolved:
                batch_input_ids.append(input_ids)
                batch_labels.append(action_labels)

            batch_input_ids.append(input_ids)
            batch_labels.append(observation_labels)

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
        }

    dataset = load_dataset(**OmegaConf.to_container(cfg.dataset))
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    ).shuffle()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    trainer_args = SFTConfig(**OmegaConf.to_container(cfg.trainer))
    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
