import json
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils.preprocessing import preprocess_query


class IntentRegressionDataset(Dataset):
    """
     JSON-backed dataset for joint intent classification + regression.

    Expected JSON format:
    [
      {
        "text": "I ate 2 apples",
        "intent": 3,                 # int class id
        "regression_target": 126.4   # float target
      }
    ]
    """

    def __init__(
        self,
        json_path: Union[str, Path],
        tokenizer_name_or_obj: Union[str, PreTrainedTokenizerBase],
        max_length: int = 128,
        text_field: str = "text",
        intent_field: str = "intent",
        regression_field: str = "regression_target",
        preprocess_text: bool = True,
    ) -> None:
        self.json_path = Path(json_path)
        self.max_length = max_length
        self.text_field = text_field
        self.intent_field = intent_field
        self.regression_field = regression_field
        self.preprocess_text = preprocess_text

        if isinstance(tokenizer_name_or_obj, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_obj)
        else:
            self.tokenizer = tokenizer_name_or_obj

        self.samples = self._load_samples(self.json_path)

    def _load_samples(self, json_path: Path) -> List[Dict[str, Any]]:
        if not json_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of records.")

        for idx, sample in enumerate(data):
            if self.text_field not in sample:
                raise ValueError(
                    f"Missing text field '{self.text_field}' in record index {idx}."
                )
            if self.intent_field not in sample:
                raise ValueError(
                    f"Missing intent field '{self.intent_field}' in record index {idx}."
                )
            if self.regression_field not in sample:
                raise ValueError(
                    f"Missing regression field '{self.regression_field}' in record index {idx}."
                )

        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        raw_text = str(sample[self.text_field])
        text = raw_text
        food_entities: List[str] = []
        if self.preprocess_text:
            proc = preprocess_query(text)
            text = str(proc["cleaned_text"]) or text
            food_entities = list(proc["food_entities"])
        intent_value = sample[self.intent_field]
        regression_value = sample[self.regression_field]

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Squeeze batch dimension because tokenizer returns shape [1, seq_len]
        item: Dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "intent_label": torch.tensor(intent_value, dtype=torch.long),
            "regression_target": torch.tensor(regression_value, dtype=torch.float),
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
        item["food_entity_count"] = torch.tensor(len(food_entities), dtype=torch.long)
        item["raw_text"] = raw_text

        if "nutrition_vector" in sample and isinstance(sample["nutrition_vector"], list):
            item["nutrition_vector"] = torch.tensor(sample["nutrition_vector"], dtype=torch.float32)
        if "calorie_budget" in sample and sample["calorie_budget"] is not None:
            item["constraint_vector"] = torch.tensor(
                [float(sample["calorie_budget"])],
                dtype=torch.float32,
            )

        return item


def create_intent_regression_dataloader(
    json_path: Union[str, Path],
    tokenizer_name_or_obj: Union[str, PreTrainedTokenizerBase],
    batch_size: int = 16,
    max_length: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    text_field: str = "text",
    intent_field: str = "intent",
    regression_field: str = "regression_target",
    preprocess_text: bool = True,
) -> DataLoader:
    dataset = IntentRegressionDataset(
        json_path=json_path,
        tokenizer_name_or_obj=tokenizer_name_or_obj,
        max_length=max_length,
        text_field=text_field,
        intent_field=intent_field,
        regression_field=regression_field,
        preprocess_text=preprocess_text,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    # Minimal sanity-check usage example
    # Update "sample_data.json" and tokenizer name as needed.
    loader = create_intent_regression_dataloader(
        json_path="sample_data.json",
        tokenizer_name_or_obj="distilbert-base-uncased",
        batch_size=2,
    )

    first_batch = next(iter(loader))
    print({k: v.shape for k, v in first_batch.items()})
