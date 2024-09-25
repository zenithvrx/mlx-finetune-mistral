import os
from datasets import load_dataset


def create_chat(sample):
    system_message = """You are an advanced AI assistant specialized in translating
    natural language questions into SQL queries. Your primary function is to interpret
    user inquiries about data and generate accurate SQL queries based on the provided
    database schema. SCHEMA: {schema}
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": system_message.format(schema=sample["context"]),
            },
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
    }


def prepare_dataset(hf_dataset_name, num_samples=5000, test_size=0.15, val_size=0.15):
    dataset = load_dataset(hf_dataset_name, split="train")
    dataset = dataset.shuffle(seed=1337).select(range(num_samples))
    dataset = dataset.map(create_chat, remove_columns=dataset.features, batched=False)

    train_test = dataset.train_test_split(test_size=test_size + val_size)
    test_val = train_test["test"].train_test_split(
        test_size=val_size / (test_size + val_size)
    )

    return {
        "train": train_test["train"],
        "test": test_val["test"],
        "validation": test_val["train"],
    }


def save_dataset(dataset, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    dataset["train"].to_json(
        os.path.join(output_folder, "train.jsonl"), orient="records", lines=True
    )
    dataset["test"].to_json(
        os.path.join(output_folder, "test.jsonl"), orient="records", lines=True
    )
    dataset["validation"].to_json(
        os.path.join(output_folder, "validation.jsonl"), orient="records", lines=True
    )


if __name__ == "__main__":
    hf_dataset_name = "b-mc2/sql-create-context"
    output_folder = "chat_dataset"

    dataset = prepare_dataset(hf_dataset_name, 5000)
    save_dataset(dataset, output_folder)

    print(f"Dataset sizes:")
    print(f"Train: {len(dataset['train'])}")
    print(f"Test: {len(dataset['test'])}")
    print(f"Validation: {len(dataset['validation'])}")
    print(f"\nSample train item:")
    print(dataset["train"][0]["messages"])
