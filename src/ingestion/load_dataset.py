from datasets import load_dataset


DATASET_NAME = "ai4bharat/MSMARCO-XI"


def load_msmarco_xi():
    print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded successfully!\n")
    print(dataset)

    return dataset


if __name__ == "__main__":
    dataset = load_msmarco_xi()
