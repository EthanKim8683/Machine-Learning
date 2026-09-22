from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
import hydra
from datasets import load_dataset


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

    ds = load_dataset(**OmegaConf.to_container(cfg.dataset, resolve=True))


if __name__ == "__main__":
    main()
