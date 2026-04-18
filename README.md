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

## TTS (Piper, Offline)

This project uses Piper for offline text-to-speech in `app/services/tts_service.py`.

1. Install Python dependencies:
```powershell
pip install -r requirements.txt
```

2. Install Piper runtime (choose one option):

Option A (system binary):
```bash
sudo apt update
sudo apt install piper-tts
```

Option B (Python package in virtual environment):
```powershell
pip install piper-tts
```

3. Download the selected voice model (example below uses `en_GB-cori-high`):
```bash
mkdir -p models/piper
cd models/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/cori/high/en_GB-cori-high.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/cori/high/en_GB-cori-high.onnx.json
```

4. Optional environment variables for Piper tuning:
```env
PIPER_MODEL_PATH=models/piper/en_GB-cori-high.onnx
PIPER_CONFIG_PATH=models/piper/en_GB-cori-high.onnx.json
PIPER_SPEAKER_ID=0
PIPER_SPEED=1.16
PIPER_NOISE_SCALE=0.58
PIPER_NOISE_W=0.76
PIPER_SENTENCE_SILENCE=0.04
```
