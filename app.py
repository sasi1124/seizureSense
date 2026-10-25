import os
import tempfile
import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from seizure_detection import SeizureDetector

APP_NAME = "SeizureSense"
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
LOGO_PATH = os.path.join(ASSETS, "logo_crop.png")
if not os.path.exists(LOGO_PATH):
    LOGO_PATH = os.path.join(ASSETS, "logo.png")
HAS_LOGO = os.path.exists(LOGO_PATH)

st.set_page_config(
    page_title=APP_NAME,
    page_icon=LOGO_PATH if HAS_LOGO else "🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, .stButton button, label {
        font-family: 'Inter', sans-serif;
        font-size: 18px;
    }
    .block-container { padding-top: 2rem; max-width: 1400px; }
    h1 { font-size: 3rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    h3 { font-size: 1.5rem !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
    }
    [data-testid="stSidebar"] * { font-size: 1.05rem; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 10px 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    [data-testid="stMetricLabel"] { justify-content: center; font-size: 0.85rem; opacity: 0.8; }
    [data-testid="stMetricLabel"] p { font-size: 0.85rem; }
    [data-testid="stMetricValue"] { justify-content: center; font-size: 1.5rem; font-weight: 700; }
    [data-testid="stAlert"] {
        border-radius: 14px;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 1rem 1.2rem;
    }
    .stButton button {
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
    }
    [data-testid="stImage"] img, video { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def annotate(img, flag):
    color = (0, 0, 255) if flag else (0, 180, 0)
    text = "ALERT: possible seizure activity" if flag else "Monitoring"
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 36), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, text, (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def read_frames(cap):
    """Yield BGR frames from an opened cv2.VideoCapture."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


class LiveProcessor:
    def __init__(self):
        self.detector = SeizureDetector()
        self.score = 0.0
        self.prob = 0.0
        self.flag = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        drawn, self.flag, self.score = self.detector.process(img)
        self.prob = self.detector.prob
        return av.VideoFrame.from_ndarray(annotate(drawn, self.flag), format="bgr24")


ACCOUNT_NAME = "Account Name"
EMERGENCY_CONTACT = "John Doe"


def show_score(score_box, chart_box, history):
    """Show the latest VSViG score and a line chart of the score over time."""
    s = np.array(history) if history else np.zeros(1)
    first = int(np.argmax(s >= 0.5)) if (s >= 0.5).any() else None
    with score_box.container():
        c = st.columns(3)
        c[0].metric("Current score", f"{s[-1]:.2f}")
        c[1].metric("Peak", f"{s.max():.2f}")
        c[2].metric("Average", f"{s.mean():.2f}")
        c = st.columns(2)
        c[0].metric("Frames ≥ 0.5", f"{100 * (s >= 0.5).mean():.0f}%")
        c[1].metric("First alert", f"frame {first}" if first is not None else "None")
    if len(history) > 1:
        chart_box.line_chart(history, height=220)


def show_result(status_box, flag):
    if flag:
        status_box.error("Seizure Detected")
        st.session_state["alerted"] = True
    else:
        status_box.success("Seizure Not Detected")
    # Once triggered, the message stays until the user clears it
    if st.session_state.get("alerted"):
        footer_box.markdown(
            f"<p style='text-align:center;font-size:1.2rem;color:#ff4b4b'>"
            f"<b>Contacting {EMERGENCY_CONTACT}...</b></p>",
            unsafe_allow_html=True,
        )


# ---------------- Sidebar ----------------
with st.sidebar:
    if HAS_LOGO:
        st.image(LOGO_PATH, use_container_width=True)
    st.subheader("Upload Video File")
    up = st.file_uploader(
        "Upload Video File",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
    )
    live = st.toggle("Live Video")
    if st.button("Clear alert"):
        st.session_state["alerted"] = False
    st.markdown("<br>" * 8, unsafe_allow_html=True)
    st.markdown(f"**{ACCOUNT_NAME}**")

# Reset the alert when a new video is uploaded (or the upload is removed)
upload_id = (getattr(up, "file_id", None) or (up.name, up.size)) if up else None
if st.session_state.get("upload_id") != upload_id:
    st.session_state["upload_id"] = upload_id
    st.session_state["alerted"] = False

# ---------------- Header ----------------
st.title(APP_NAME)

# ---------------- Image + score/status ----------------
img_col, info_col = st.columns([3, 2], gap="large")

# Bottom message placeholder (only filled when a seizure is detected)
footer_box = st.empty()

# Status, score metrics and chart sit beside the video, so they stay visible while it plays
with info_col:
    status_box = st.empty()
    show_result(status_box, False)
    score_box = st.empty()
    chart_box = st.empty()

with img_col:
    if live:
        ctx = webrtc_streamer(
            key="neuroshield-live",
            video_processor_factory=LiveProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1920, "min": 1280},
                    "height": {"ideal": 1080, "min": 720},
                    "frameRate": {"ideal": 30},
                },
                "audio": False,
            },
            video_html_attrs={"style": {"width": "100%"}, "controls": False, "autoPlay": True},
            async_processing=True,
        )
        # Reset the alert each time the live stream is (re)started
        if ctx.state.playing and not st.session_state.get("was_playing"):
            st.session_state["alerted"] = False
            footer_box.empty()
        st.session_state["was_playing"] = ctx.state.playing
    elif up:
        analyze = st.button("Analyze video", type="primary")
        view, bar = st.empty(), st.progress(0)
        if analyze:
            st.session_state["alerted"] = False
            footer_box.empty()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(up.read())
            cap = cv2.VideoCapture(tmp.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            detector = SeizureDetector()
            detected = False
            history = []
            for i, frame in enumerate(read_frames(cap)):
                drawn, flag, _ = detector.process(frame, t=i / fps)
                detected = detected or flag
                history.append(detector.prob)
                if i % 3 == 0:
                    shown = annotate(drawn, detected)
                    view.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
                    show_result(status_box, detected)
                if i % 15 == 0:
                    show_score(score_box, chart_box, history)
                bar.progress(min(i / total, 1.0))
            cap.release()
            bar.progress(1.0)
            show_result(status_box, detected)
            show_score(score_box, chart_box, history)
    else:
        st.info("Upload a video or turn on Live Video.")

# Keep score/status updating while the live stream is playing
if live and ctx.state.playing:
    live_history = []
    while ctx.state.playing:
        proc = ctx.video_processor
        if proc is not None:
            live_history = (live_history + [proc.prob])[-300:]
            show_result(status_box, proc.flag)
            show_score(score_box, chart_box, live_history)
        time.sleep(0.3)

