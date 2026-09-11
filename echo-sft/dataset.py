from pathlib import Path
from datasets import load_dataset, Dataset


SEED = 42
TRAJECTORIES_PER_INSTANCE = 3
DATASET_PATH = Path("data/my-SWE-smith-trajectories.parquet")


def main():
    dataset = load_dataset("SWE-bench/SWE-smith-trajectories", split="xml")
    dataset = dataset.to_pandas()

    dataset = dataset.sample(frac=1, random_state=SEED)
    dataset = dataset[dataset["resolved"]]
    dataset = dataset[dataset["model"].str.startswith("claude-3-7")]
    dataset = dataset.groupby("instance_id").head(TRAJECTORIES_PER_INSTANCE)
    dataset = dataset.sample(frac=1, random_state=SEED)

    dataset = Dataset.from_pandas(dataset, preserve_index=False)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH)


if __name__ == "__main__":
    main()
