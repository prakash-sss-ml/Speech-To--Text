"""Live microphone transcription for Streamlit Community Cloud."""
import os
import queue
import threading

import av
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from streamlit_webrtc import WebRtcMode, webrtc_streamer

SAMPLE_RATE = 16000


def setting(name, default):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except FileNotFoundError:
        return os.getenv(name, default)


@st.cache_resource
def load_model(size, device, compute_type):
    return WhisperModel(size, device=device, compute_type=compute_type), threading.Lock()


class AudioInbox:
    """WebRTC callbacks only enqueue audio; Streamlit owns UI and inference."""
    def __init__(self):
        self.frames = queue.Queue(maxsize=3000)
        self.overflow = threading.Event()

    def receive(self, frame):
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            self.overflow.set()
        return frame


st.set_page_config(page_title="Live Speech-to-Text", page_icon="🎙️")
st.title("Live Speech-to-Text")
st.caption("Continuous microphone transcription with Whisper. Text appears after each audio chunk.")

size = str(setting("MODEL_SIZE", "base"))
device = str(setting("DEVICE", "cpu"))
compute_type = str(setting("COMPUTE_TYPE", "int8"))
try:
    chunk_seconds = float(setting("CHUNK_SECONDS", 3.5))
    if not 0.5 <= chunk_seconds <= 30:
        raise ValueError("CHUNK_SECONDS must be between 0.5 and 30")
except ValueError as exc:
    st.error(str(exc))
    st.stop()
chunk_samples = int(SAMPLE_RATE * chunk_seconds)
with st.spinner("Loading speech model (first run downloads model files)..."):
    model, model_lock = load_model(size, device, compute_type)

if "audio_inbox" not in st.session_state:
    st.session_state.audio_inbox = AudioInbox()
    st.session_state.audio_buffer = np.empty(0, dtype=np.float32)
    st.session_state.resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
    st.session_state.transcript = ""
    st.session_state.was_playing = False
    st.session_state.last_language = None

languages = {"Auto-detect": None, "English": "en", "Tamil": "ta", "Korean": "ko", "Hindi": "hi"}
label = st.selectbox("Language (choose before starting)", list(languages))
st.caption(f"Model: {size} · Device: {device} · Precision: {compute_type} · Chunk: {chunk_seconds:g} seconds")
ice_servers = setting("ICE_SERVERS", [{"urls": ["stun:stun.l.google.com:19302"]}])
ctx = webrtc_streamer(
    key="live-microphone",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration={"iceServers": list(ice_servers)},
    media_stream_constraints={"video": False, "audio": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": True}},
    audio_frame_callback=st.session_state.audio_inbox.receive,
    async_processing=False,
)
st.caption("Press START and allow microphone access. Press STOP and wait for remaining audio to finish.")


@st.fragment(run_every=0.5)
def update_transcript():
    state = st.session_state
    inbox = state.audio_inbox
    playing = ctx.state.playing
    if playing and not state.was_playing:
        state.last_language = languages[label]
    state.was_playing = playing
    incoming = []
    # Bound each update even while new microphone frames arrive.
    for _ in range(3000):
        try:
            frame = inbox.frames.get_nowait()
        except queue.Empty:
            break
        for converted in state.resampler.resample(frame):
            incoming.append(converted.to_ndarray().reshape(-1))
    if incoming:
        state.audio_buffer = np.concatenate([state.audio_buffer, *incoming])
    if state.audio_buffer.size > SAMPLE_RATE * 60:
        state.audio_buffer = state.audio_buffer[:SAMPLE_RATE * 60]
        inbox.overflow.set()
    if inbox.overflow.is_set():
        st.warning("Audio was dropped because processing could not keep up. Stop and let pending audio finish; try a smaller model for this server.")
    buffer = state.audio_buffer
    if buffer.size >= chunk_samples or (not playing and buffer.size):
        count = min(chunk_samples, buffer.size)
        audio = buffer[:count].copy()
        try:
            if count >= SAMPLE_RATE // 3 and np.sqrt(np.mean(audio * audio)) >= 0.004:
                with model_lock:
                    segments, _ = model.transcribe(
                        audio, language=state.last_language, beam_size=1,
                        vad_filter=True, condition_on_previous_text=False,
                        no_speech_threshold=0.6,
                    )
                    text = " ".join(s.text.strip() for s in segments).strip()
                if text:
                    state.transcript = " ".join(filter(None, [state.transcript, text]))
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
        finally:
            state.audio_buffer = buffer[count:]
    pending = state.audio_buffer.size / SAMPLE_RATE
    if playing:
        st.info(f"Listening · {pending:.1f} seconds buffered")
    elif pending or not inbox.frames.empty():
        st.info(f"Finishing transcription · {pending:.1f} seconds buffered. Wait before restarting.")
    else:
        st.caption("Ready")
        state.resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
    if pending > 15:
        st.warning("Transcription is falling behind live audio on this server.")
    st.text_area("Transcript", value=state.transcript, height=260, disabled=True)
    st.download_button("Download transcript", state.transcript, file_name="transcript.txt", mime="text/plain")
    if st.button("Clear transcript", disabled=playing or pending > 0):
        state.transcript = ""
        inbox.overflow.clear()
        st.rerun(scope="fragment")


update_transcript()
st.caption("Audio is sent to the hosting server for transcription, not saved as an audio file. Fixed chunk boundaries may split words.")

