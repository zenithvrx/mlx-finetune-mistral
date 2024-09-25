import json
from pathlib import Path
from typing import Dict, List

from transformers import PreTrainedTokenizer


class Dataset:
    """
    Light-weight wrapper to hold a dataset.
    """

    def __init__(self, data: List[Dict[str, str]], text_key: str = "text"):
        self._text_key = text_key
        self._data = data

    def __getitem__(self, idx: int):
        return self._data[idx][self._text_key]

    def __len__(self):
        if self._data is None:
            return 0
        return len(self._data)


class ChatDataset(Dataset):
    """
    A dataset for chat data in the format of {"messages": [...]}
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(self, data: List[Dict[str, str]], tokenizer: PreTrainedTokenizer):
        super().__init__(data)
        self._tokenizer = tokenizer

    def __getitem__(self, idx: int):
        messages = self._data[idx]["messages"]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text


class CompletionsDataset(Dataset):
    """
    A dataset for prompt-completion data in the format of {"prompt": ..., "completion": ...}
    or using user-provided keys for prompt and completion values
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        completion_key: str = "completion",
    ):
        super().__init__(data)
        self._tokenizer = tokenizer
        self._prompt_key = prompt_key
        self._completion_key = completion_key

    def __getitem__(self, idx: int):
        data = self._data[idx]
        text = self._tokenizer.apply_chat_template(
            [
                {"role": "user", "content": data[self._prompt_key]},
                {"role": "assistant", "content": data[self._completion_key]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text


def create_dataset(path: Path, tokenizer: PreTrainedTokenizer = None):
    if not path.exists():
        return []
    with open(path, "r") as fid:
        data = [json.loads(l) for l in fid]
    if "messages" in data[0]:
        return ChatDataset(data, tokenizer)
    elif "prompt" in data[0] and "completion" in data[0]:
        return CompletionsDataset(data, tokenizer)
    elif "text" in data[0]:
        return Dataset(data)
    else:
        raise ValueError(
            "Unsupported data format, check the supported formats here:\n"
            "https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data."
        )


def load_dataset(args, tokenizer: PreTrainedTokenizer):
    names = ("train", "validation", "test")
    data_path = Path(args.data)

    train, valid, test = [
        create_dataset(data_path / f"{n}.jsonl", tokenizer) for n in names
    ]

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test
