from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Ответ эндпоинта GET /health — проверка работоспособности."""

    status: str = Field(
        ...,
        description="Статус сервиса. 'ok' — сервис работает",
        examples=["ok"],
    )
    model_loaded: bool = Field(
        ...,
        description="Загружена ли ML-модель в память",
        examples=[True],
    )


class PredictRequest(BaseModel):
    """Запрос эндпоинта POST /predict — один текст на классификацию."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Текст на русском языке для классификации",
        examples=["Сегодня прекрасная погода"],
    )


class PredictResponse(BaseModel):
    """Ответ эндпоинта POST /predict — класс и уверенность."""

    label: str = Field(
        ...,
        description="Класс текста: 'toxic' или 'neutral'",
        examples=["neutral"],
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Уверенность модели в диапазоне [0.0, 1.0]",
        examples=[0.9876],
    )


class BatchRequest(BaseModel):
    """Запрос эндпоинта POST /predict_batch — список текстов."""

    texts: list[str] = Field(
        ...,
        description="Список текстов для пакетной классификации",
        examples=[["Отличная работа!", "Фу. Очень плохо..."]],
    )


class BatchItem(BaseModel):
    """
    Один результат в пакетном ответе.
    Как PredictResponse, но плюс поле text с исходным текстом
    """

    text: str = Field(
        ...,
        description="Исходный текст",
    )
    label: str = Field(
        ...,
        description="Класс текста: 'toxic' или 'neutral'",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Уверенность модели в диапазоне [0.0, 1.0]",
    )


class BatchResponse(BaseModel):
    """Ответ эндпоинта POST /predict_batch."""

    results: list[BatchItem] = Field(
        ...,
        description="Список результатов в том же порядке, что и входные тексты",
    )


class ModelInfoResponse(BaseModel):
    """Ответ эндпоинта GET /model_info — метаданные модели."""

    model_name: str = Field(
        ...,
        description="Идентификатор модели на HuggingFace",
        examples=["s-nlp/russian_toxicity_classifier"],
    )
    task: str = Field(
        ...,
        description="Тип ML-задачи",
        examples=["text-classification"],
    )
    labels: list[str] = Field(
        ...,
        description="Возможные классы, которые возвращает модель",
        examples=[["neutral", "toxic"]],
    )
    transformers_version: str = Field(
        ...,
        description="Версия библиотеки transformers",
        examples=["4.45.0"],
    )
    torch_version: str = Field(
        ...,
        description="Версия библиотеки torch",
        examples=["2.4.0"],
    )
    loaded: bool = Field(
        ...,
        description="Загружена ли модель в память прямо сейчас",
        examples=[True],
    )
