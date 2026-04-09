# Final Fixed FYP

Organized FastAPI version of the smart mirror project.

Authentication is Google OAuth-based. Manual username/password registration and login are disabled.

## Structure
- `app/` application code
- `app/agent/` LangGraph assistant logic
- `app/services/` Gmail, Google OAuth, and TTS helpers
- `app/ml/` runtime face model code
- `web/templates/` Jinja templates
- `web/static/` frontend assets
- `data/` SQLite DB and runtime data files
- `models/` ML weights
- `config/` Google credential files

## Run
```powershell
python run.py
```

Or:
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
