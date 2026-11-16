# Live Speech-to-Text

Continuous browser microphone transcription using faster-whisper. The Streamlit entry point is `streamlit_app.py`. The original FastAPI application remains in `main.py`.

## Run Streamlit locally

Use Python 3.11, install `requirements.txt`, then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the displayed localhost URL, choose a language, press START, and allow microphone access. Speak for at least 3.5 seconds. Press STOP and wait for pending transcription before restarting. Download the transcript using the download button.

## Deploy to Streamlit Community Cloud

1. Push the project to a GitHub repository. Do not upload `venv`, `.env`, or real secrets.
2. In Community Cloud, create an app and select your repository and branch.
3. Set the main file path to `streamlit_app.py` and select Python 3.11 in advanced settings.
4. Copy `.streamlit/secrets.toml.example` into the app's Secrets settings. The defaults are Whisper `base`, CPU, `int8`, and 3.5-second chunks.
5. Configure `ICE_SERVERS` with working TURN relay credentials from your provider for reliable remote audio connections. The default public STUN server alone may not work on Community Cloud or restricted networks. Provider charges may apply. Keep credentials in Secrets, never GitHub.
6. Deploy. First startup downloads the model; open the app and test microphone START, live text, STOP/final text, and download from an external network.

Official guides:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
- https://whitphx.github.io/streamlit-webrtc/deployment/

For local configuration, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and edit it. Streamlit settings read secrets first, then environment variables. `.env.example` is a reference and is not automatically loaded.

## Audio flow and limitations

Browser microphone -> WebRTC -> session audio queue -> mono 16 kHz resampling -> fixed audio chunks -> Whisper -> transcript. The model is cached across Streamlit reruns and shared with an inference lock. Audio and transcript buffers belong to each browser session. A periodic fragment processes one chunk per update while WebRTC keeps capturing audio.

Stop drains received frames and transcribes the remaining partial chunk. Extremely short or quiet audio is skipped. Fixed chunks do not overlap and may split words. This is near-real-time chunk transcription, not word-by-word streaming. Cloud CPU performance may cause a backlog, particularly with multiple users; the UI shows buffered audio and queue overflow. The incoming queue is bounded and drops new frames if overloaded. Keep sessions short and stop if the server falls behind. The transcript persists only for the browser session; download it before leaving.

Microphone audio is transmitted to the cloud server for inference and is not intentionally saved as an audio file. The model may download from Hugging Face. WebRTC requires microphone permission and HTTPS (provided by Community Cloud) or localhost.

## Original FastAPI app

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. This entry point reads environment variables, not Streamlit Secrets. The existing `/health` and `/ws/transcribe` endpoints remain available in this mode.
