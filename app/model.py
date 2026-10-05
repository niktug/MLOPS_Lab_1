"""
Обёртка над ML-моделью классификации токсичности.

Модель: s-nlp/russian_toxicity_classifier
Задача: text-classification (neutral / toxic)
"""

import torch
import transformers
from transformers import pipeline


class ToxicityModel:
    """Обёртка над моделью классификации токсичности."""

    t = 0
    MODEL_NAME = "s-nlp/russian_toxicity_classifier"
    TASK = "text-classification"
    LABELS = ["neutral", "toxic"]

    def __init__(self):
        self.classifier = None

    def load(self):
        """Загружает модель в память. Вызывается один раз при старте."""
        self.classifier = pipeline(
            self.TASK,
            model=self.MODEL_NAME,
        )

    @property
    def is_loaded(self):
        """Загружена ли модель."""
        return self.classifier is not None

    def predict(self, text):
        """Классифицирует один текст. Возвращает {label, score}."""
        result = self.classifier(text)[0]
        return {
            "label": result["label"].lower(),
            "score": round(float(result["score"]), 4),
        }

    def predict_batch(self, texts):
        """Классифицирует список текстов. Возвращает список {text, label, score}."""
        if not texts:
            return []

        results = self.classifier(texts)
        return [
            {
                "text": text,
                "label": r["label"].lower(),
                "score": round(float(r["score"]), 4),
            }
            for text, r in zip(texts, results)
        ]

    def get_info(self):
        """Метаданные модели для /model_info."""
        return {
            "model_name": self.MODEL_NAME,
            "task": self.TASK,
            "labels": self.LABELS,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "loaded": self.is_loaded,
        }
