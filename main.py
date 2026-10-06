from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://job-tracker-frontend-heqb.onrender.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

custom_stop_words = list(ENGLISH_STOP_WORDS) + [
    "experience", "looking", "developer", "background", "role", "years",
    "ability", "working", "work", "ahead", "strong", "team", "teams",
    "join", "help", "build", "building", "great", "ideal", "preferred",
    "required", "requirements", "responsibilities", "including", "new",
    "ambitious", "ambiguous", "world", "age", "apply", "applied",
]
@app.get("/")
def read_root():
    return {"message": "Hello from Kokavi Insights"}

class MatchRequest(BaseModel):
    resume_text: str
    job_description: str
def get_ai_suggestions(resume_text, job_description, missing_keywords):
    if not missing_keywords:
        return "Your resume already covers the key terms in this job description well."

    prompt = f"""A candidate's resume is being compared to a job description.
Missing keywords found: {', '.join(missing_keywords)}

Resume: {resume_text}

Job Description: {job_description}

In 3-4 concise bullet points, explain why these missing keywords matter for this role, and suggest how the candidate might naturally incorporate relevant experience into their resume if they have it. Do not fabricate experience the candidate doesn't have."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def analyze_requirements(resume_text, job_description):
    prompt = f"""You are screening a resume against a job description.

Step 1: From the job description, list the concrete requirements: skills, tools, technologies, qualifications, certifications and key responsibilities. Ignore company names, product names, benefits, culture statements and equal-opportunity or legal text. Return at most 20, ordered from most to least important. Keep each name short (2 to 6 words), like a skill, tool or qualification, not a full sentence. Mark each "must" or "nice".

Step 2: For each requirement, decide whether the resume shows it. Treat synonyms and abbreviations as a match (for example JS = JavaScript). If it is shown, copy a short exact quote from the resume as evidence. If the resume does not show it, leave evidence empty. Never invent evidence.

Return JSON only, in this shape:
{{"requirements": [{{"name": "...", "importance": "must", "evidence": "..."}}]}}

Resume:
{resume_text[:12000]}

Job description:
{job_description[:12000]}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.choices[0].message.content)["requirements"]


@app.post("/match")
def match_resume(data: MatchRequest):
    # Baseline: keyword similarity (also the fallback if the AI step fails)
    vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
    vectors = vectorizer.fit_transform([data.resume_text, data.job_description])
    tfidf_score = round(cosine_similarity(vectors[0], vectors[1])[0][0] * 100, 2)

    feature_names = vectorizer.get_feature_names_out()
    resume_vector = vectors[0].toarray()[0]
    jd_vector = vectors[1].toarray()[0]
    candidates = [
        (jd_vector[i], feature_names[i])
        for i in range(len(feature_names))
        if jd_vector[i] > 0 and resume_vector[i] == 0
    ]
    candidates.sort(reverse=True)
    missing_keywords = [word for _, word in candidates[:15]]
    matched_keywords = []
    score = tfidf_score

    try:
        requirements = analyze_requirements(data.resume_text, data.job_description)
        resume_norm = " ".join(data.resume_text.lower().split())
        weight_total = 0
        weight_matched = 0
        matched, missing = [], []
        for req in requirements:
            weight = 2 if req.get("importance") == "must" else 1
            evidence = " ".join((req.get("evidence") or "").lower().split())
            found = bool(evidence) and evidence in resume_norm  # reject invented quotes
            weight_total += weight
            if found:
                weight_matched += weight
                matched.append(req["name"])
            else:
                missing.append(req["name"])
        if weight_total:
            score = round(weight_matched / weight_total * 100, 2)
            missing_keywords = missing[:15]
            matched_keywords = matched
    except Exception:
        pass  # keep the keyword-based result

    ai_suggestions = get_ai_suggestions(
        data.resume_text, data.job_description, missing_keywords
    )

    return {
        "match_score_percent": score,
        "missing_keywords": missing_keywords,
        "matched_keywords": matched_keywords,
        "keyword_similarity_percent": tfidf_score,
        "ai_suggestions": ai_suggestions,
    }