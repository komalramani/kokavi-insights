# kokavi-insights

A small API that compares a resume with a job description. It finds the job's real requirements, checks which ones the resume shows, and suggests what to improve.

**Live API:** https://kokavi-insights.onrender.com (interactive docs at `/docs`)

It is used by the Match Checker in my [Job Application Tracker](https://job-tracker-frontend-heqb.onrender.com).

## How it works

1. **Read the job.** An AI model (OpenAI gpt-4o-mini) lists up to 20 concrete requirements from the job description: skills, tools, qualifications and responsibilities. Each is marked "must" or "nice". Company names, benefits and legal text are ignored.
2. **Check the resume.** For each requirement, the model must quote the line in the resume that shows it. If the quote is not really in the resume, it does not count. This stops the AI from inventing matches.
3. **Score.** The score is the share of requirements the resume covers, with "must" items counting double.
4. **Suggest.** The missing requirements are sent to the model again to explain why they matter and how to show relevant experience.
5. **Fallback.** If the AI step fails, the API falls back to a simple keyword-similarity (TF-IDF) score so it still returns a result.

## API

`POST /match`

Request:
```json
{ "resume_text": "...", "job_description": "..." }
```

Response:
```json
{
  "match_score_percent": 53.33,
  "missing_keywords": ["Python programming", "..."],
  "matched_keywords": ["..."],
  "keyword_similarity_percent": 18.51,
  "ai_suggestions": "..."
}
```

`GET /` returns a simple hello message.

## Tech

Python, FastAPI, scikit-learn (TF-IDF and cosine similarity), OpenAI API, deployed on Render.

## Run locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your key (never commit it):
```
OPENAI_API_KEY=your-key-here
```

Start the server:
```bash
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs.

## Known limitations

- The score is an estimate. The AI can pick a slightly different set of requirements each run, so the score can vary between runs. The missing list and suggestions are more useful than the exact percentage.
- The quote check is strict, so a real match can sometimes be counted as missing.
- Each check makes two AI calls, so it takes about 10 to 20 seconds.
- It runs on Render's free tier. After 15 minutes idle it goes to sleep, so the first request can take 30 to 60 seconds.
- It only reads pasted text, not uploaded PDF or Word files.

## Ideas for next

- Make the quote check more forgiving and the score steadier.
- Show matched requirements in the tracker UI.
- Accept resume file uploads.