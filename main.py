from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os
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
    "experience", "looking", "developer", "background", "role", "years"
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
@app.post("/match")
def match_resume(data: MatchRequest):
    documents = [data.resume_text, data.job_description]

    vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = round(float(similarity[0][0]) * 100, 2)

    feature_names = vectorizer.get_feature_names_out()
    resume_vector = tfidf_matrix[0].toarray()[0]
    jd_vector = tfidf_matrix[1].toarray()[0]

    missing_keywords = [
        feature_names[i] for i in range(len(feature_names))
        if jd_vector[i] > 0 and resume_vector[i] == 0
    ]

    ai_suggestions = get_ai_suggestions(data.resume_text, data.job_description, missing_keywords)

    return {
        "match_score_percent": score,
        "missing_keywords": missing_keywords,
        "ai_suggestions": ai_suggestions
    }