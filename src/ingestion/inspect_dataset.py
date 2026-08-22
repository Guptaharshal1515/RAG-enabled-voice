from datasets import load_dataset


DATASET_NAME = "ai4bharat/MSMARCO-XI"


def inspect_dataset():
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    dataset = load_dataset(DATASET_NAME)
    print(dataset)

    for split_name, split_data in dataset.items():
        print("\n" + "=" * 70)
        print(f"SPLIT: {split_name}")
        print("=" * 70)

        print(f"\nNumber of records: {len(split_data)}")

        print("\nFeatures / Schema:")
        print(split_data.features)

        print("\nColumn names:")
        print(split_data.column_names)

        print("\nFirst example:")
        print(split_data[0])

        print("\nFirst 3 examples:")
        for i in range(min(3, len(split_data))):
            print(f"\n--- Example {i + 1} ---")
            print(split_data[i])


if __name__ == "__main__":
    inspect_dataset()
