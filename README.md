# Fine-tuning Mistral models using LoRA on Apple MLX

MLX is an array framework for machine learning on Apple silicon. This repo shows how to fine-tune a Mistral 7B Instruct model with low rank adaptation (LoRA) locally on an Apple device.

## Contents

- [Setup](#Setup)
  - [Libraries](#Libraries)
  - [Model](#Model)
- [Run](#Run)
  - [Data](#Data)
  - [Fine-tune](#Fine-tune)
- [Results](#Results)

## Setup

### Libraries

Set up a Python virtual environment:

```shell
python3 -m venv venv
source venv/bin/activate
```

Install the required libraries:

```shell
pip install mlx mlx-lm transformers datasets "huggingface_hub[cli]"
```

### Model

To download the non-quantized Mistral 7B Instruct v2 model from Hugging Face, run:

```shell
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2
```

To view where the model was downloaded, run:

```shell
ls -la ~/.cache/huggingface/hub
```

Verify that the base model runs locally by running:

```shell
mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.2 --prompt "hi, who are you?"
```

## Run

### Data

#### Source

This project uses the `b-mc2/sql-create-context` dataset from Hugging Face. It contains 78,577 examples, each comprising of:

- A natural language query
- An SQL CREATE TABLE statement
- An SQL query answering the question using the CREATE statement as context

#### Processing

The `create_data.py` script prepares the dataset for fine-tuning. It does the following:

1. Downloads the dataset from Hugging Face
2. Extracts a specified number of samples for fine-tuning
3. Splits the data into training, validation, and test sets
4. Saves the processed data as .jsonl files

#### Usage

To prepare the dataset, run:

```
python create_data.py
```

#### Output

After running the script, you'll find the following files in the `chat_dataset/` directory:

- `train.jsonl`: Training data
- `validation.jsonl`: Validation data
- `test.jsonl`: Test data

These files contain the processed dataset ready for use in fine-tuning.

#### Customization

If you need to adjust the number of samples or modify the split ratios, you can edit the parameters in `create_data.py`.

### Fine-tune

The fine-tuning script is `lora.py`.

To start the fine-tuning process, run:

```
python lora.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --train \
    --iters 1000 \
    --batch-size 1 \
    --lora-layers 4 \
    --data chat_dataset
```

Fine-tuning a Mistral 7B model on 1000 iterations will take roughly 5-10 minutes on a M-series chip. During fine-tuning, adapter weights are saved in `adapters.npz`.

#### Fusing

After fine-tuning, fuse the LoRA weights with the base model to create a single, optimized model for inference.

Run the fuse script with:
```
python fuse.py --model mistralai/Mistral-7B-Instruct-v0.2 --save-path fused_model --adapter-file adapters.npz
```
This will save the fused model in the `fused_model/` directory.

### Results

Finally, run inference locally with the newly fine-tuned Mistral model in `fused_model/`.

```bash
mlx_lm.generate --model fused_model --max-tokens 1000 --prompt "You are a text to SQL query translator. Generate a SQL query based on the provided SCHEMA and QUESTION.

SCHEMA:
CREATE TABLE products (
product_id INTEGER PRIMARY KEY,
product_name VARCHAR(100),
category VARCHAR(50),
price DECIMAL(10, 2),
stock_quantity INTEGER
);

QUESTION: Find the names and prices of all products in the 'Electronics' category with a price less than 500 and a stock quantity greater than 10. Order the results by price in descending order."
```