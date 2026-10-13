import os

# Must be set before importing main, because main creates the OpenAI client on import
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_docs_page_loads():
    response = client.get("/docs")
    assert response.status_code == 200


def test_match_rejects_missing_fields():
    response = client.post("/match", json={})
    assert response.status_code == 422


def test_match_rejects_missing_job_description():
    response = client.post("/match", json={"resume_text": "Python developer"})
    assert response.status_code == 422