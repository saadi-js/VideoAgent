import os
import tempfile
from html import escape

import streamlit as st
from dotenv import load_dotenv

from core.rag_engine import ask_question
from main import run_pipeline, get_source_cache_key


load_dotenv()

st.set_page_config(
    page_title="VidAgent | Meeting Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #080808;
        --panel: #111111;
        --panel-2: #171717;
        --yellow: #f5d547;
        --yellow-bright: #ffe66d;
        --muted: #9c9c9c;
        --line: #2b2b2b;
    }

    .stApp {
        background:
            radial-gradient(circle at 90% 4%, rgba(245, 213, 71, .12), transparent 25rem),
            radial-gradient(circle at 8% 30%, rgba(245, 213, 71, .05), transparent 22rem),
            var(--ink);
        color: #f5f5f5;
        font-family: 'Space Grotesk', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    .block-container { max-width: 1380px; padding: 2rem 3rem 4rem; }

    .brand {
        display: flex; justify-content: space-between; align-items: center;
        margin: .5rem 0 3rem; padding-bottom: 1.3rem;
        border-bottom: 1px solid var(--line);
    }
    .brand-name { color: var(--yellow); font-size: 1.25rem; font-weight: 700; letter-spacing: .16em; }
    .brand-subtitle { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .08em; }
    .status { color: var(--yellow); font-family: 'DM Mono', monospace; font-size: .75rem; }
    .status-dot { display: inline-block; width: 7px; height: 7px; margin-right: 7px; border-radius: 50%; background: var(--yellow); box-shadow: 0 0 14px var(--yellow); }

    .hero-kicker { color: var(--yellow); font-family: 'DM Mono', monospace; font-size: .75rem; letter-spacing: .18em; text-transform: uppercase; }
    .hero-title { margin: .45rem 0 .6rem; font-size: clamp(2.4rem, 5vw, 5rem); line-height: .96; letter-spacing: -.06em; font-weight: 700; }
    .hero-copy { max-width: 620px; color: var(--muted); font-size: 1.02rem; line-height: 1.6; }

    .input-card, .metric-card, .content-card, .chat-card {
        background: linear-gradient(145deg, rgba(28,28,28,.96), rgba(13,13,13,.96));
        border: 1px solid var(--line); border-radius: 18px;
        box-shadow: 0 18px 50px rgba(0,0,0,.2);
    }
    .input-card { padding: 1.35rem; margin: 2rem 0; border-top: 2px solid var(--yellow); }
    .section-label { color: var(--yellow); font-family: 'DM Mono', monospace; font-size: .7rem; letter-spacing: .15em; text-transform: uppercase; margin-bottom: .65rem; }
    .metric-card { min-height: 128px; padding: 1.2rem; }
    .metric-value { color: #fff; font-size: 1.15rem; font-weight: 600; margin-top: .65rem; }
    .metric-copy { color: var(--muted); font-size: .82rem; line-height: 1.45; }
    .content-card { padding: 1.4rem; margin-top: 1.2rem; min-height: 180px; }
    .content-card p, .content-card li { color: #d3d3d3; line-height: 1.65; }
    .chat-card { padding: 1.4rem; margin-top: 1.2rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(28,28,28,.96), rgba(13,13,13,.96));
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: 0 18px 50px rgba(0,0,0,.18);
        padding: .25rem;
    }
    .insight-card {
        min-height: 160px;
    }
    .transcript-pane, .chat-pane {
        min-height: 650px;
    }
    .pane-heading {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding-bottom: .8rem;
        margin-bottom: .9rem;
        border-bottom: 1px solid var(--line);
    }
    .pane-title {
        color: #fff;
        font-size: 1.15rem;
        font-weight: 600;
    }
    .pane-meta {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: .68rem;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .transcript-text {
        height: 565px;
        overflow-y: auto;
        padding: .2rem .35rem .2rem .1rem;
        color: #d3d3d3;
        white-space: pre-wrap;
        line-height: 1.7;
        font-size: .91rem;
    }
    .insight-copy {
        color: #d3d3d3;
        line-height: 1.55;
        font-size: .88rem;
    }
    .insight-copy p, .insight-copy li { margin-bottom: .45rem; }
    .chat-pane [data-testid="stChatMessage"] {
        margin-bottom: .65rem;
    }
    .chat-pane [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        margin-left: 2.5rem;
        background: rgba(245, 213, 71, .1);
        border-color: rgba(245, 213, 71, .35);
    }
    .chat-pane [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        margin-right: 2.5rem;
        background: #151515;
    }
    .chat-form {
        border-top: 1px solid var(--line);
        margin-top: 1rem;
        padding-top: 1rem;
    }
    .chat-form [data-testid="stFormSubmitButton"] button {
        min-height: 2.45rem;
    }
    .chat-empty {
        min-height: 430px;
        display: grid;
        place-items: center;
        color: var(--muted);
        text-align: center;
        padding: 2rem;
    }

    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
        background: #0c0c0c; color: #fff; border: 1px solid #3a3a3a; border-radius: 10px;
    }
    div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--yellow); box-shadow: 0 0 0 1px var(--yellow);
    }
    .stButton > button {
        background: var(--yellow); color: #0b0b0b; border: 0; border-radius: 9px;
        font-weight: 700; min-height: 2.7rem; transition: all .2s ease;
    }
    .stButton > button:hover { background: var(--yellow-bright); color: #000; transform: translateY(-1px); }
    [data-testid="stFileUploader"] section { background: #0c0c0c; border: 1px dashed #555; border-radius: 10px; }
    [data-testid="stChatMessage"] { background: #151515; border: 1px solid var(--line); border-radius: 13px; }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "result": None,
        "source": None,
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_insight_card(label: str, content: str, icon: str) -> None:
    with st.container(border=True):
        st.markdown(
            f'<div class="section-label">{icon} &nbsp; {label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(content or "_No information returned._")


initialize_state()

st.markdown(
    """
    <div class="brand">
        <div><div class="brand-name">VIDAGENT</div><div class="brand-subtitle">MEETING INTELLIGENCE / LOCAL-FIRST RAG</div></div>
        <div class="status"><span class="status-dot"></span>SYSTEM READY</div>
    </div>
    <div class="hero-kicker">Turn long-form video into decisions</div>
    <div class="hero-title">Listen less.<br><span style="color:#f5d547">Understand more.</span></div>
    <div class="hero-copy">Drop in a YouTube meeting or local recording. VidAgent transcribes, summarizes, indexes, and lets you interrogate the conversation.</div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">01 / SOURCE</div>', unsafe_allow_html=True)
source_url = st.text_input(
    "YouTube URL",
    placeholder="https://youtube.com/watch?v=...",
    label_visibility="collapsed",
)
uploaded_file = st.file_uploader(
    "Or upload a local audio/video file",
    type=["mp3", "mp4", "m4a", "wav", "webm", "mov", "ogg"],
)
process_clicked = st.button("PROCESS MEETING  →", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if process_clicked:
    if not source_url.strip() and uploaded_file is None:
        st.error("Add a YouTube URL or upload a local media file.")
    else:
        temporary_path = None
        source = source_url.strip()
        try:
            if uploaded_file is not None and not source:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
                    file.write(uploaded_file.getbuffer())
                    temporary_path = file.name
                source = temporary_path

            with st.status("Processing meeting...", expanded=True) as status:
                st.write("Acquiring audio and checking local cache...")
                result = run_pipeline(source)
                status.update(label="Meeting ready", state="complete", expanded=False)

            st.session_state.result = result
            st.session_state.source = source
            st.session_state.messages = []
            st.rerun()
        except Exception as error:
            st.error(f"Processing failed: {error}")
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)


result = st.session_state.result
if result:
    st.markdown('<div class="section-label">02 / SIGNAL REPORT</div>', unsafe_allow_html=True)
    title = result.get("title", "Untitled meeting")
    summary = result.get("summary", "")
    transcript = result.get("transcript", "")
    cols = st.columns(4)
    metrics = [
        ("◈", "MEETING", title),
        ("◌", "TRANSCRIPT", f"{len(transcript.split()):,} words"),
        ("✦", "ACTION ITEMS", "Extracted"),
        ("◎", "RAG INDEX", "Online"),
    ]
    for column, (icon, label, value) in zip(cols, metrics):
        with column:
            st.markdown(
                f'<div class="metric-card"><div class="section-label">{icon} &nbsp; {label}</div><div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown('<div class="section-label">EXECUTIVE SUMMARY</div>', unsafe_allow_html=True)
        st.markdown(summary or "_No summary returned._")

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">KEY SIGNALS</div>', unsafe_allow_html=True)
    insight_columns = st.columns(3, gap="medium")
    insight_cards = [
        ("→", "ACTION ITEMS", result.get("action_items", "")),
        ("◆", "DECISIONS", result.get("decisions", "")),
        ("?", "OPEN QUESTIONS", result.get("questions", "")),
    ]
    for column, (icon, label, content) in zip(insight_columns, insight_cards):
        with column:
            render_insight_card(label, content, icon)

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">03 / WORKSPACE</div>', unsafe_allow_html=True)
    transcript_column, chat_column = st.columns([1.08, .92], gap="large")

    with transcript_column:
        with st.container(border=True):
            st.markdown(
                '<div class="pane-heading"><span class="pane-title">Transcript</span>'
                f'<span class="pane-meta">{len(transcript.split()):,} words</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="transcript-text">{escape(transcript or "No transcript returned.")}</div>',
                unsafe_allow_html=True,
            )

    with chat_column:
        with st.container(border=True):
            st.markdown(
                '<div class="pane-heading"><span class="pane-title">Ask the meeting</span>'
                '<span class="pane-meta">Context-aware chat</span></div>',
                unsafe_allow_html=True,
            )
            with st.container(height=500, border=False):
                if st.session_state.messages:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                else:
                    st.markdown(
                        '<div class="chat-empty">Ask about a decision, action item, speaker, '
                        'or any detail in the meeting.</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="chat-form">', unsafe_allow_html=True)
            with st.form("meeting_chat_form", clear_on_submit=True):
                prompt = st.text_input(
                    "Ask a question about this meeting",
                    placeholder="Ask about a decision, action item, speaker, or detail...",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("SEND  →", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if submitted and prompt.strip():
                st.session_state.messages.append(
                    {"role": "user", "content": prompt.strip()}
                )
                with st.spinner("Searching the meeting..."):
                    session_id = get_source_cache_key(st.session_state.source)
                    answer = ask_question(
                        result["rag_chain"],
                        prompt.strip(),
                        session_id=session_id,
                    )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.rerun()
else:
    st.markdown(
        '<div class="content-card" style="text-align:center; padding:3rem 1rem;"><div class="section-label">WAITING FOR SOURCE</div><div style="font-size:1.35rem; font-weight:600;">Your meeting report will appear here.</div><div style="color:#9c9c9c; margin-top:.5rem;">Audio, transcript, embeddings, and insights in one vivid workspace.</div></div>',
        unsafe_allow_html=True,
    )
