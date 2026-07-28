"""API contract tests.

The pipeline is replaced with a stub, so these exercise routing, validation and
serialisation without loading 3 GB of weights.
"""

import pytest
from fastapi.testclient import TestClient

from moderation import api
from moderation.pipeline import ModerationPipeline


class StubClassifier:
    def __init__(self, label, score=0.99):
        self.label, self.score = label, score

    def __call__(self, text):
        if isinstance(text, list):
            return [{"label": self.label, "score": self.score} for _ in text]
        return [{"label": self.label, "score": self.score}]


@pytest.fixture
def client(monkeypatch):
    """A client whose lifespan installs a stubbed pipeline."""
    stub = ModerationPipeline(
        hate_clf=StubClassifier("Offensive"), toxicity_clf=StubClassifier("LABEL_0")
    )
    monkeypatch.setattr(api, "ModerationPipeline", lambda *a, **k: stub)
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_reports_ok_once_models_are_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_moderate_returns_a_full_verdict(client):
    response = client.post("/moderate", json={"text": "انت غبي"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "Offensive"
    assert payload["action"] == "mask"
    assert payload["is_harmful"] is True
    assert payload["masked_text"] is not None
    assert payload["explanation"]


def test_empty_text_is_rejected_by_validation(client):
    assert client.post("/moderate", json={"text": ""}).status_code == 422


def test_missing_field_is_rejected(client):
    assert client.post("/moderate", json={}).status_code == 422


def test_overlong_text_is_rejected(client):
    assert client.post("/moderate", json={"text": "x" * 5000}).status_code == 422


def test_batch_returns_one_verdict_per_input(client):
    response = client.post("/moderate/batch", json={"texts": ["a", "b", "c"]})

    assert response.status_code == 200
    assert len(response.json()) == 3


def test_empty_batch_is_rejected(client):
    assert client.post("/moderate/batch", json={"texts": []}).status_code == 422


def test_oversized_batch_is_rejected(client):
    response = client.post("/moderate/batch", json={"texts": ["x"] * 65})

    assert response.status_code == 422
