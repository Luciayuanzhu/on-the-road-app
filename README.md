# Live Companion Backend

Minimal Railway-ready backend for the live AI companion MVP.

## What it implements

- `deviceId` to `userId` anonymous identity resolution
- persisted user settings
- session creation plus websocket live session handling
- backend-owned prompt and response construction
- mic-off session support through location and text events
- explicit session-scoped camera assist via `visual_assist.start` and `visual_assist.stop`
- bookmarks, editable notes, transcript excerpts, and async bookmark summaries

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m companion_backend
```

The app listens on `PORT` or `8000` and exposes `GET /health`.

## Environment

Use `.env.example` as the baseline:

- `DATABASE_URL`
- `PUBLIC_BASE_URL`
- `PORT`
- `SUMMARY_DELAY_SECONDS`
- `LOCATION_RESPONSE_COOLDOWN_SECONDS`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `GOOGLE_PLACES_API_KEY`
- `TICKETMASTER_API_KEY`
- `GEMINI_BASE_URL`
- `GOOGLE_PLACES_BASE_URL`
- `TICKETMASTER_BASE_URL`
- `GOOGLE_PLACES_LANGUAGE_CODE`
- `GEMINI_LIVE_MODEL`
- `GEMINI_LIVE_WS_URL`
- `GEMINI_LIVE_AUDIO_IDLE_SECONDS`
- `TICKETMASTER_EVENT_CACHE_SECONDS`

If `DATABASE_URL` is omitted, the app creates a local SQLite database at `./data/companion.db`.

## Test

```bash
PYTHONPATH=src pytest -q
```
