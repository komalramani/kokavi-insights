# KokaVi Insights

I built this to answer a question I kept asking while applying for jobs: how well does my resume actually match this job description, and what am I missing?

You paste in a resume and a job description. It gives back a match score, a list of keywords the job asks for that your resume doesn't mention, and a few plain-English suggestions on how to deal with the gaps.

It's also my first real Python project. I come from a teaching and research background and have mostly built in JavaScript and React, so I wanted a Python project that did something useful rather than another calculator.

## What you get back

- **A match score.** I use TF-IDF and cosine similarity from scikit-learn. 0 means nothing in common, 100 means identical text.
- **Missing keywords.** Words that show up in the job description but nowhere in the resume.
- **Suggestions from an LLM.** For the missing keywords, `gpt-4o-mini` explains why they matter for the role and how you could mention related experience, if you really have it. I told it not to make anything up, because a tool that invents qualifications isn't helping anyone.

## Quick example

Request:

```json
{
  "resume_text": "Experienced React and Node.js developer with PostgreSQL background",
  "job_description": "Looking for a React developer with Docker, Kubernetes, and AWS experience"
}
```

Response (suggestions shortened):

```json
{
  "match_score_percent": 12.74,
  "missing_keywords": ["aws", "docker", "kubernetes"],
  "ai_suggestions": "- These three tools are central to the role... - If you've used any of them, even on a team project, say so..."
}
```

The score is low because the only shared word is "React", which is the right result for that pair.

## Running it yourself

You'll need Python 3 and an OpenAI API key.

```bash
git clone https://github.com/komalramani/kokavi-insights.git
cd kokavi-insights
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn scikit-learn openai python-dotenv
```

Make a file called `.env` in the project folder:

```
OPENAI_API_KEY=your_key_here
```

It's in `.gitignore`, so it won't get committed. Then start the server:

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs and you can try `POST /match` from the Swagger page without writing any code.

## Things worth knowing

- TF-IDF only matches exact words. "JS" and "JavaScript" count as different terms, and it has no idea about synonyms. Treat the score as a rough signal, not a verdict.
- With only two documents, the word weighting is crude. A real version would be tuned on many job descriptions.
- Each request makes one OpenAI call, which costs a fraction of a cent on `gpt-4o-mini`.
- I added a few custom stop words ("experience", "looking", "role", "years") because job postings are full of them and they showed up as fake "missing keywords".

## What I'm doing next

- Add a "Check Match" button to my [Job Application Tracker](https://github.com/komalramani/job-tracker)
- Deploy this to Render
- Add a `requirements.txt` and some tests
- Try embeddings instead of TF-IDF so synonyms count

## Built with

Python, FastAPI, Pydantic, scikit-learn, the OpenAI API, python-dotenv.

Komal Ramani
