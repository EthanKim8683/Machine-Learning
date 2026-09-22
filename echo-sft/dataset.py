from pathlib import Path
from datasets import load_dataset, Dataset


SEED = 42
TRAJECTORIES_PER_INSTANCE = 3
DATASET_PATH = Path("data/my-SWE-smith-trajectories.parquet")
INSTANCE_TYPE_ORDER = [
    "pr_mirror",
    "lm_rewrite",
    "procedural",
    "combine",
    "lm_modify",
    "other",
]


def parse_instance_type(instance_id):
    if ".pr_" in instance_id:
        return "pr_mirror"
    if ".lm_rewrite" in instance_id:
        return "lm_rewrite"
    if ".lm_modify" in instance_id:
        return "lm_modify"
    if ".combine" in instance_id:
        return "combine"
    if ".func" in instance_id:
        return "procedural"
    return "other"


def main():
    dataset = load_dataset("SWE-bench/SWE-smith-trajectories", split="xml")
    dataset = dataset.to_pandas()

    dataset = dataset.sample(frac=1, random_state=SEED)
    dataset = dataset[dataset["resolved"]]
    dataset = dataset[dataset["model"].str.startswith("claude-3-7")]
    dataset = dataset.groupby("instance_id", sort=False).head(TRAJECTORIES_PER_INSTANCE)

    dataset["instance_type"] = dataset["instance_id"].map(parse_instance_type)
    order = {e: i for i, e in enumerate(INSTANCE_TYPE_ORDER)}
    dataset = dataset.sort_values("instance_type", key=lambda column: column.map(order))
    dataset = dataset.drop(columns=["instance_type"])

    dataset = Dataset.from_pandas(dataset, preserve_index=False)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH)


if __name__ == "__main__":
    main()
