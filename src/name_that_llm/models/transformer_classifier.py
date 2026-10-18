"""PyTorch / Hugging Face Transformer fine-tuning module with class-weighted loss."""

import copy
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from name_that_llm.config import MODEL_TO_IDX, NUM_CLASSES, TARGET_MODELS
from name_that_llm.metrics import compute_balanced_accuracy


def get_device() -> torch.device:
    """Select the best available compute device (MPS on Apple Silicon, CUDA on Nvidia, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for pre-tokenized passages."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer,
        max_length: int = 384,
    ):
        encodings = tokenizer(
            [str(t) for t in texts],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.token_type_ids = encodings.get("token_type_ids", None)
        self.labels = (
            torch.tensor(labels, dtype=torch.long) if labels is not None else None
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class TransformerClassifier:
    """End-to-end wrapper around HuggingFace sequence classification models with balanced loss."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        num_classes: int = NUM_CLASSES,
        max_length: int = 384,
        batch_size: int = 8,
        lr: float = 2e-5,
        epochs: int = 3,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device or get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_classes,
            dtype=torch.float32,
        ).to(self.device)

    def fit(self, train_texts: List[str], train_labels: List[int], val_texts=None, val_labels=None):
        """Fine-tune the transformer backbone with inverse-frequency class weights."""
        # Compute balanced class weights to optimize directly for Balanced Accuracy
        classes, counts = np.unique(train_labels, return_counts=True)
        total_samples = len(train_labels)
        class_weights = np.ones(self.num_classes, dtype=np.float32)
        for cls, count in zip(classes, counts):
            class_weights[cls] = total_samples / (self.num_classes * count)

        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, max_length=self.max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_bal_acc = -1.0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["labels"].to(self.device)

                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

                outputs = self.model(**kwargs)
                loss = loss_fn(outputs.logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            if val_texts is not None and val_labels is not None:
                val_preds = self.predict(val_texts)
                val_bal_acc = compute_balanced_accuracy(val_labels, val_preds)
                print(
                    f"  Epoch {epoch + 1}/{self.epochs} | Train Loss: {avg_loss:.4f} | Val Bal Acc: {val_bal_acc:.4f}"
                )
                if val_bal_acc > best_bal_acc:
                    best_bal_acc = val_bal_acc
                    best_state = copy.deepcopy(self.model.state_dict())
            else:
                print(f"  Epoch {epoch + 1}/{self.epochs} | Train Loss: {avg_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  [Best Model Restored] Peak Val Bal Acc: {best_bal_acc:.4f}")

        if self.device.type == "mps":
            torch.mps.empty_cache()

        return self

    def predict_proba(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Predict class probability distributions."""
        self.model.eval()
        bs = batch_size or (self.batch_size * 2)
        dataset = TextClassificationDataset(texts, None, self.tokenizer, max_length=self.max_length)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)
                outputs = self.model(**kwargs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        if self.device.type == "mps":
            torch.mps.empty_cache()

        return np.vstack(all_probs)

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict target class indices."""
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=-1)

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

