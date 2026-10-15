import os

# Must be set before importing main, because main creates the OpenAI client on import.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi.testclient import TestClient

import main
from main import app

client = TestClient(app)

RESUME = "Built REST APIs with Python and FastAPI. Wrote Jest tests."
JOB = "Python developer needing Kubernetes and Terraform experience."


def fake_requirements(reqs):
    """Return a stand-in for analyze_requirements that gives a fixed answer."""
    return lambda resume_text, job_description: reqs


def raise_error(*args, **kwargs):
    raise RuntimeError("OpenAI is down")


# ---------- Route tests ----------

def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_docs_page_loads():
    assert client.get("/docs").status_code == 200


def test_match_rejects_missing_fields():
    assert client.post("/match", json={}).status_code == 422


def test_match_rejects_missing_job_description():
    assert client.post("/match", json={"resume_text": "Python developer"}).status_code == 422


# ---------- Logic tests (OpenAI mocked) ----------

def test_score_weights_must_twice_as_much_as_nice(monkeypatch):
    reqs = [
        {"name": "Python", "importance": "must", "evidence": "Built REST APIs with Python"},
        {"name": "Docker", "importance": "nice", "evidence": ""},
    ]
    monkeypatch.setattr(main, "analyze_requirements", fake_requirements(reqs))
    monkeypatch.setattr(main, "get_ai_suggestions", lambda *a, **k: "ok")

    data = client.post("/match", json={"resume_text": RESUME, "job_description": JOB}).json()

    # "must" matched = weight 2, "nice" missing = weight 1 -> 2 / 3 = 66.67%
    assert data["match_score_percent"] == 66.67
    assert data["matched_keywords"] == ["Python"]
    assert data["missing_keywords"] == ["Docker"]


def test_all_requirements_matched_scores_100(monkeypatch):
    reqs = [
        {"name": "Python", "importance": "must", "evidence": "Built REST APIs with Python"},
        {"name": "Jest", "importance": "nice", "evidence": "Wrote Jest tests"},
    ]
    monkeypatch.setattr(main, "analyze_requirements", fake_requirements(reqs))
    monkeypatch.setattr(main, "get_ai_suggestions", lambda *a, **k: "ok")

    data = client.post("/match", json={"resume_text": RESUME, "job_description": JOB}).json()

    assert data["match_score_percent"] == 100.0
    assert data["missing_keywords"] == []


def test_invented_evidence_quote_is_rejected(monkeypatch):
    # The AI claims a quote that is NOT in the resume, so it must count as missing.
    reqs = [
        {"name": "Kubernetes", "importance": "must", "evidence": "5 years of Kubernetes in production"},
    ]
    monkeypatch.setattr(main, "analyze_requirements", fake_requirements(reqs))
    monkeypatch.setattr(main, "get_ai_suggestions", lambda *a, **k: "ok")

    data = client.post("/match", json={"resume_text": RESUME, "job_description": JOB}).json()

    assert data["match_score_percent"] == 0.0
    assert data["missing_keywords"] == ["Kubernetes"]
    assert data["matched_keywords"] == []


def test_falls_back_to_tfidf_when_ai_fails(monkeypatch):
    monkeypatch.setattr(main, "analyze_requirements", raise_error)
    monkeypatch.setattr(main, "get_ai_suggestions", lambda *a, **k: "ok")

    data = client.post("/match", json={"resume_text": RESUME, "job_description": JOB}).json()

    # Score falls back to the keyword-similarity score, and JD-only words are reported missing.
    assert data["match_score_percent"] == data["keyword_similarity_percent"]
    assert "kubernetes" in data["missing_keywords"]
    assert data["matched_keywords"] == []


def test_ai_suggestions_are_returned_in_response(monkeypatch):
    monkeypatch.setattr(main, "analyze_requirements", fake_requirements([]))
    monkeypatch.setattr(main, "get_ai_suggestions", lambda *a, **k: "Add Docker experience.")

    data = client.post("/match", json={"resume_text": RESUME, "job_description": JOB}).json()

    assert data["ai_suggestions"] == "Add Docker experience."


def test_no_missing_keywords_skips_openai_call():
    # With nothing missing, get_ai_suggestions returns a fixed message without calling OpenAI.
    result = main.get_ai_suggestions(RESUME, JOB, [])
    assert result == "Your resume already covers the key terms in this job description well."



def test_evidence_with_section_label_is_accepted(monkeypatch):
    resume = "Skills: Python, FastAPI, REST API design, PostgreSQL."
    reqs = [{"name": "PostgreSQL", "importance": "must", "evidence": "Skills: PostgreSQL"}]
    monkeypatch.setattr(main, "analyze_requirements", fake_requirements(reqs))
    monkeypatch.setattr(main, "get_ai_suggestions", lambda *a, **k: "ok")

    data = client.post("/match", json={"resume_text": resume, "job_description": JOB}).json()

    assert data["match_score_percent"] == 100.0
    assert data["matched_keywords"] == ["PostgreSQL"]


def test_evidence_naming_a_skill_not_in_resume_is_rejected(monkeypatch):
    resume = "Skills: Python, FastAPI, REST API design, PostgreSQL."
    reqs = [{"name": "Docker", "importance": "must", "evidence": "Skills: Docker"}]
    monkeypatch.setattr(main, "analyze_requirements", fake_requirements(reqs))
    monkeypatch.setattr(main, "get_ai_suggestions", lambda *a, **k: "ok")

    data = client.post("/match", json={"resume_text": resume, "job_description": JOB}).json()

    assert data["match_score_percent"] == 0.0
    assert data["missing_keywords"] == ["Docker"]