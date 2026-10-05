import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """
    Тестовый клиент FastAPI — клиент создаётся один раз на весь модуль тестов
    """
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """/health — проверка работоспособности сервиса."""

    def test_returns_200_with_ok_status_and_loaded_model(self, client):
        """
        Эндпоинт должен вернуть 200, статус 'ok' и флаг model_loaded=True.
        """
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


class TestPredictEndpoint:
    """/predict — классификация одного текста."""

    def test_valid_text_returns_label_and_score(self, client):
        """
        Валидный текст должен вернуть ровно два поля: label и score.
        - ровно 2 поля в ответе
        - label — только 'toxic' или 'neutral'
        - score — float в диапазоне [0.0, 1.0]
        """
        response = client.post("/predict", json={"text": "Сегодня прекрасная погода"})

        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {"label", "score"}
        assert body["label"] in ("toxic", "neutral")
        assert isinstance(body["score"], float)
        assert 0.0 <= body["score"] <= 1.0

    def test_empty_text_rejected_with_422(self, client):
        """
        Защита от невалидного ввода, статус 422.
        """
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422


class TestPredictBatchEndpoint:
    """/predict_batch — классификация списка текстов."""

    def test_processes_multiple_texts_in_order(self, client):
        """
        Пакет из трёх текстов должен вернуть три результата в том же порядке.
        - 200 OK
        - length(results) == length(texts)
        - порядок сохранён: text в каждом результате совпадает с исходным
        - каждый результат — валидная схема: text, label, score.
        """
        payload = {
            "texts": [
                "Отличная работа!",
                "Фу. Очень плохо.",
                "Сдача лабораторки.",
            ]
        }
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "results" in body
        assert len(body["results"]) == 3

        for original, result in zip(payload["texts"], body["results"]):
            assert set(result.keys()) == {"text", "label", "score"}
            assert result["text"] == original
            assert result["label"] in ("toxic", "neutral")
            assert 0.0 <= result["score"] <= 1.0

    def test_empty_list_returns_empty_results(self, client):
        """
        Пустой список — валидный ввод, сервис отвечает 200 и пустым results.
        """
        response = client.post("/predict_batch", json={"texts": []})
        assert response.status_code == 200
        body = response.json()
        assert body["results"] == []
