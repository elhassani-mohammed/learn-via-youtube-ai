"""
YouTube Chat Assistant
----------------------
Search YouTube, watch a video, and chat about it with a Groq LLM that knows the
video only through its transcript (injected behind the scenes).

Run:  streamlit run app.py
Files: app.py + locales.json (all UI text lives in locales.json)
"""

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import requests
import streamlit as st
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi

# --------------------------------------------------------------------------- #
# i18n: every user-facing string comes from locales.json
# --------------------------------------------------------------------------- #
LOCALES = json.loads(Path(__file__).with_name("locales.json").read_text(encoding="utf-8"))
DEFAULT_LANG = "en"


def detect_browser_lang() -> str:
    """Pick the UI language from the browser's Accept-Language header."""
    try:
        header = st.context.headers.get("Accept-Language", "")
    except Exception:
        header = ""
    for part in header.split(","):
        code = part.split(";")[0].strip().lower()[:2]
        if code in LOCALES:
            return code
    return DEFAULT_LANG


if "lang" not in st.session_state:
    st.session_state.lang = detect_browser_lang()


def t(key: str, **kwargs) -> str:
    text = LOCALES.get(st.session_state.lang, {}).get(key) or LOCALES[DEFAULT_LANG].get(key, key)
    return text.format(**kwargs) if kwargs else text


st.set_page_config(page_title=t("app_title"), page_icon="🎬", layout="wide")

if st.session_state.lang == "ar":
    st.markdown(
        "<style>[data-testid='stChatMessage'], [data-testid='stSidebar'] {direction: rtl;}</style>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
AUTHOR_NAME = "Mohammed El Hassani"  # shown in the footer
NON_CHAT_MODEL_HINTS = ("whisper", "tts", "guard", "orpheus", "embed", "playai", "compound", "safeguard")
MAX_OUTPUT_TOKENS = 1500
# One model answers every question. Default: the first of these available on the user's key;
# otherwise the available model closest to 70B.
DEFAULT_MODEL_PREFERENCES = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant")
FULL_CONTEXT_CHAR_LIMIT = 20_000
RETRIEVAL_CHAR_BUDGET = 12_000
CHUNK_SIZE_CHARS = 900
HISTORY_TURNS = 8
# Study session: (task id, minutes). Display names come from locales.json (task_<id>).
FOCUS_TASKS = (("listening", 20), ("shadowing", 15), ("speaking", 15), ("writing", 10))
FOCUS_IDS = [tid for tid, _ in FOCUS_TASKS]

# Tune these two if your screen needs it: space (px) taken by everything that is NOT the video / chat.
LAYOUT_CSS = """
<style>
:root {
    --stage-offset: 320px;  /* px used on the page by everything that is NOT the video (search bar, title, focus panel) */
    --chat-chrome: 133px;   /* px used inside the chat panel by its title + input box */
    /* one height shared by the video and the chat panel; the video width follows it (16:9) */
    --stage-h: min(calc(100vh - var(--stage-offset)), calc(62cqw * 9 / 16));
}
.block-container, [data-testid="stMainBlockContainer"] {
    container-type: inline-size; max-width: 100% !important;
    padding: 2.6rem 1.2rem 1.6rem 1.2rem !important; }
[data-testid="stColumn"]:has(.st-key-video_box) {
    flex: 0 0 auto !important; width: calc(var(--stage-h) * 16 / 9) !important; min-width: 0 !important; }
[data-testid="stColumn"]:has(.st-key-chat_panel) { flex: 1 1 0 !important; width: auto !important; min-width: 0 !important; }
.st-key-video_box iframe, .st-key-video_box video {
    width: 100% !important; height: var(--stage-h) !important; border-radius: 12px; }
/* Streamlit gives these containers "flex: 1 1 0%", which makes the browser ignore `height`; flex:none restores it. */
.st-key-chat_panel { flex: 0 0 auto !important; height: var(--stage-h) !important; box-sizing: border-box;
    overflow: hidden !important; border-radius: 12px; }
.st-key-chat_msgs, [data-testid="stLayoutWrapper"]:has(> .st-key-chat_msgs) {
    flex: 0 0 auto !important; height: calc(var(--stage-h) - var(--chat-chrome)) !important; max-height: none !important; }
.app-footer { text-align: center; font-size: .8rem; opacity: .7; padding: .8rem 0 .2rem 0; }
.app-footer.pinned { position: absolute; left: 0; right: 0; bottom: 4px; padding: 0; pointer-events: none; z-index: 1; }
[data-testid="stElementContainer"]:has(.app-footer.pinned) { position: absolute; }  /* no flow space, so no layout shift */
.focus-clock { font-size: 1.3rem; font-weight: 700; white-space: nowrap; }
.pane-title { font-size: 1.05rem; font-weight: 700; }
.vid-title { font-weight: 600; line-height: 1.3; margin: .1rem 0 .3rem 0; }
</style>
"""
SEARCH_PAGE_SIZE = 12


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_count(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return ""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n / 1_000:.1f}K".replace(".0K", "K")
    return str(int(n))


def parse_iso_duration(iso: str) -> int:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    h, mi, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + s


def stamp_to_seconds(stamp: str) -> int:
    parts = [int(p) for p in stamp.split(":")]
    total = 0
    for p in parts:
        total = total * 60 + p
    return total


def escape_md(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+\-.!~|<>$])", r"\\\1", text)


def truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def extract_video_id(text: str) -> str | None:
    text = text.strip()
    m = re.search(r"(?:v=|/videos/|embed/|youtu\.be/|/shorts/|/live/)([A-Za-z0-9_-]{11})", text)
    if m:
        return m.group(1)
    # bare 11-char ID: require a digit, "-" or "_" so ordinary words aren't mistaken for IDs
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", text) and re.search(r"[0-9_-]", text):
        return text
    return None


# --------------------------------------------------------------------------- #
# YouTube: search, metadata, transcript
# --------------------------------------------------------------------------- #
def thumb_url(video_id: str) -> str:
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


@st.cache_data(show_spinner=False, ttl=600)
def yt_search_api(query: str, api_key: str, page_token: str | None = None) -> tuple[list[dict], str | None]:
    """YouTube Data API v3: search.list + videos.list (for duration and views)."""
    params = {"part": "snippet", "type": "video", "maxResults": SEARCH_PAGE_SIZE, "q": query, "key": api_key}
    if page_token:
        params["pageToken"] = page_token
    r = requests.get("https://www.googleapis.com/youtube/v3/search", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    ids = [it["id"]["videoId"] for it in data.get("items", []) if it.get("id", {}).get("videoId")]
    details = {}
    if ids:
        r2 = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "snippet,contentDetails,statistics", "id": ",".join(ids), "key": api_key},
            timeout=15,
        )
        r2.raise_for_status()
        details = {it["id"]: it for it in r2.json().get("items", [])}
    results = []
    for vid in ids:
        d = details.get(vid, {})
        sn = d.get("snippet", {})
        results.append({
            "id": vid,
            "title": sn.get("title", ""),
            "channel": sn.get("channelTitle", ""),
            "views": d.get("statistics", {}).get("viewCount"),
            "duration": parse_iso_duration(d.get("contentDetails", {}).get("duration", "")),
            "thumb": thumb_url(vid),
        })
    return results, data.get("nextPageToken")


@st.cache_data(show_spinner=False, ttl=600)
def yt_search_ytdlp(query: str) -> list[dict]:
    """Keyless fallback search through yt-dlp."""
    from yt_dlp import YoutubeDL

    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{SEARCH_PAGE_SIZE}:{query}", download=False)
    results = []
    for e in info.get("entries", []):
        vid = e.get("id")
        if not vid or len(vid) != 11:
            continue
        results.append({
            "id": vid,
            "title": e.get("title") or "",
            "channel": e.get("channel") or e.get("uploader") or "",
            "views": e.get("view_count"),
            "duration": int(e.get("duration") or 0),
            "thumb": thumb_url(vid),
        })
    return results


def do_search(query: str, token: str | None = None, append: bool = False) -> None:
    api_key = st.session_state.get("yt_key_in") or ""
    results, next_token = None, None
    if api_key:
        try:
            results, next_token = yt_search_api(query, api_key, token)
        except Exception:
            results = None  # quota exceeded, bad key... fall back below
    if results is None:
        try:
            results, next_token = yt_search_ytdlp(query), None
        except Exception:
            st.session_state.search_error = True
            return
    if append:
        seen = {r["id"] for r in st.session_state.results}
        results = st.session_state.results + [r for r in results if r["id"] not in seen]
    st.session_state.results = results
    st.session_state.next_token = next_token
    st.session_state.query = query
    st.session_state.search_error = False


def load_more() -> None:
    do_search(st.session_state.query, st.session_state.next_token, append=True)


@st.cache_data(show_spinner=False, ttl=3600)
def get_metadata(video_id: str, api_key: str | None) -> dict:
    """Title, channel, description. Data API if a key is set, keyless oEmbed otherwise."""
    if api_key:
        try:
            r = requests.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={"part": "snippet,statistics", "id": video_id, "key": api_key},
                timeout=15,
            )
            r.raise_for_status()
            items = r.json().get("items", [])
            if items:
                sn, stats = items[0]["snippet"], items[0].get("statistics", {})
                return {
                    "title": sn.get("title", ""),
                    "channel": sn.get("channelTitle", ""),
                    "description": sn.get("description", "")[:1500],
                    "published": sn.get("publishedAt", "")[:10],
                    "views": stats.get("viewCount"),
                }
        except Exception:
            pass
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=15,
        )
        r.raise_for_status()
        d = r.json()
        return {"title": d.get("title", ""), "channel": d.get("author_name", ""),
                "description": "", "published": "", "views": None}
    except Exception:
        return {"title": "", "channel": "", "description": "", "published": "", "views": None}


@st.cache_data(show_spinner=False, ttl=3600)
def get_transcript(video_id: str) -> list[dict]:
    """Best transcript automatically: human-made first, then auto-generated, any language."""
    api = YouTubeTranscriptApi()
    available = sorted(api.list(video_id), key=lambda tr: tr.is_generated)
    fetched = available[0].fetch()
    return [{"text": s.text, "start": s.start} for s in fetched]


# --------------------------------------------------------------------------- #
# Transcript chunking + tiny BM25 retrieval
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_chunks(snippets: list[dict]) -> list[dict]:
    chunks, buf, start, size = [], [], None, 0
    for s in snippets:
        text = s["text"].replace("\n", " ").strip()
        if not text:
            continue
        if start is None:
            start = s["start"]
        buf.append(text)
        size += len(text) + 1
        if size >= CHUNK_SIZE_CHARS:
            chunks.append({"start": start, "text": " ".join(buf)})
            buf, start, size = [], None, 0
    if buf:
        chunks.append({"start": start or 0, "text": " ".join(buf)})
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1)
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {tok: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for tok, n in df.items()}

    def score(self, query: list[str], i: int) -> float:
        dl, s = len(self.docs[i]), 0.0
        for tok in query:
            f = self.tf[i].get(tok)
            if not f:
                continue
            s += self.idf.get(tok, 0) * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            )
        return s


def select_context(chunks: list[dict], question: str) -> str:
    """Whole transcript if short, otherwise the best-matching chunks in video order."""
    full = "\n".join(f"[{fmt_time(c['start'])}] {c['text']}" for c in chunks)
    if len(full) <= FULL_CONTEXT_CHAR_LIMIT:
        return full

    bm25 = BM25([tokenize(c["text"]) for c in chunks])
    q = tokenize(question)
    ranked = sorted(range(len(chunks)), key=lambda i: bm25.score(q, i), reverse=True)

    if not q or bm25.score(q, ranked[0]) == 0 or len(q) <= 3:  # broad question: sample evenly
        step = max(1, len(full) // RETRIEVAL_CHAR_BUDGET)
        ranked = list(range(0, len(chunks), step))

    picked, used = [], 0
    for i in ranked:
        cost = len(chunks[i]["text"]) + 10
        if used + cost <= RETRIEVAL_CHAR_BUDGET:
            picked.append(i)
            used += cost
    picked.sort()
    return "\n".join(f"[{fmt_time(chunks[i]['start'])}] {chunks[i]['text']}" for i in picked)


# --------------------------------------------------------------------------- #
# Groq: available models, automatic selection, streaming with fallback
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=600)
def list_chat_models(api_key: str) -> list[dict]:
    """Models actually available to this API key (no hardcoded list)."""
    client = Groq(api_key=api_key)
    models = []
    for m in client.models.list().data:
        mid = m.id
        if any(h in mid.lower() for h in NON_CHAT_MODEL_HINTS):
            continue
        if getattr(m, "active", True) is False:
            continue
        models.append({"id": mid, "ctx": int(getattr(m, "context_window", 8192) or 8192)})
    return sorted(models, key=lambda m: m["id"])


def model_size_b(model_id: str) -> float:
    """Parameter count (billions) read from the model id, e.g. '70b' -> 70. Unknown -> 30."""
    sizes = re.findall(r"(\d+(?:\.\d+)?)b", model_id.lower())
    return max(float(s) for s in sizes) if sizes else 30.0


def pick_default_model(models: list[dict]) -> str:
    available = {m["id"] for m in models}
    for preferred in DEFAULT_MODEL_PREFERENCES:
        if preferred in available:
            return preferred
    return min(models, key=lambda m: abs(math.log(model_size_b(m["id"]) / 70)))["id"]


def strip_think(pieces):
    """Remove <think>…</think> blocks that some reasoning models put in the content."""
    open_tag, close_tag = "<think>", "</think>"
    inside, buf = False, ""
    for piece in pieces:
        buf += piece
        while True:
            if inside:
                end = buf.find(close_tag)
                if end == -1:
                    buf = buf[-(len(close_tag) - 1):]
                    break
                buf, inside = buf[end + len(close_tag):], False
            else:
                start = buf.find(open_tag)
                if start == -1:
                    keep = len(open_tag) - 1
                    if len(buf) > keep:
                        yield buf[:-keep]
                        buf = buf[-keep:]
                    break
                if start:
                    yield buf[:start]
                buf, inside = buf[start + len(open_tag):], True
    if not inside and buf:
        yield buf


def build_system_prompt(meta: dict, context: str) -> str:
    return f"""You are a helpful assistant for the video the user is watching. You know the video ONLY through its transcript below.

VIDEO
Title: {meta.get('title')}
Channel: {meta.get('channel')}
Description: {meta.get('description') or 'n/a'}

TRANSCRIPT (with [mm:ss] timestamps; may be partial excerpts for long videos)
{context}

RULES
- Answer using only the transcript and metadata above. Never invent content.
- If the answer is not in the transcript, say so plainly.
- Cite moments with timestamps like [12:34] when useful.
- Reply in the same language as the user's question.
- Speak as if you watched the video; don't mention "transcript" or "excerpts" unless asked how you know.
- Be clear and well structured; use bullet points for lists."""


def stream_answer(client: Groq, candidates: list[str], system_prompt: str, history: list[dict], used: dict):
    """Try each candidate model in order until one produces output."""
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in history[-HISTORY_TURNS * 2:]
    ]
    last_error = None
    for model_id in candidates:
        started = False
        try:
            stream = client.chat.completions.create(
                model=model_id, messages=messages, temperature=0.3,
                max_tokens=MAX_OUTPUT_TOKENS, stream=True,
            )
            raw = (c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta.content)
            for piece in strip_think(raw):
                started = True
                used["model"] = model_id
                yield piece
            if started:
                return
        except Exception as e:
            if started:
                raise
            last_error = e
    raise last_error or RuntimeError("no model produced an answer")


# --------------------------------------------------------------------------- #
# State + callbacks
# --------------------------------------------------------------------------- #
def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.getenv(name, "")
    except Exception:
        return os.getenv(name, "")


for key, default in {
    "results": [], "next_token": None, "query": "", "search_error": False,
    "selected_id": None, "video": None, "load_error": None, "view": "results",
    "messages": [], "start": 0, "autoplay": False,
    # focus session
    "done": {}, "timer_task": FOCUS_IDS[0], "timer_left": FOCUS_TASKS[0][1] * 60,
    "timer_running": False, "timer_end": 0.0, "pending_done": None, "just_finished": None,
}.items():
    st.session_state.setdefault(key, default)


def select_video(video_id: str) -> None:
    st.session_state.selected_id = video_id
    st.session_state.view = "watch"


def show_results() -> None:
    st.session_state.view = "results"


def jump_to(seconds: int) -> None:
    st.session_state.start = seconds
    st.session_state.autoplay = True


def clear_chat() -> None:
    st.session_state.messages = []


# ---- focus timer ----------------------------------------------------------- #
def task_seconds(task_id: str) -> int:
    return dict(FOCUS_TASKS)[task_id] * 60


def timer_remaining() -> int:
    s = st.session_state
    if s.timer_running:
        return max(0, int(math.ceil(s.timer_end - time.time())))
    return s.timer_left


def set_task(task_id: str) -> None:
    s = st.session_state
    s.timer_task, s.timer_running, s.timer_left = task_id, False, task_seconds(task_id)


def timer_toggle() -> None:
    s = st.session_state
    if s.timer_running:
        s.timer_left, s.timer_running = timer_remaining(), False
    else:
        if s.timer_left <= 0:
            s.timer_left = task_seconds(s.timer_task)
        s.timer_end, s.timer_running = time.time() + s.timer_left, True


def timer_reset() -> None:
    s = st.session_state
    s.timer_running, s.timer_left = False, task_seconds(s.timer_task)


def timer_next() -> None:
    i = FOCUS_IDS.index(st.session_state.timer_task)
    set_task(FOCUS_IDS[(i + 1) % len(FOCUS_IDS)])


def finish_task() -> None:
    """Timer reached zero: tick the task, move on to the next unfinished one."""
    s = st.session_state
    finished = s.timer_task
    s.timer_running = False
    s.pending_done = s.just_finished = finished
    later = FOCUS_IDS[FOCUS_IDS.index(finished) + 1:] + FOCUS_IDS[:FOCUS_IDS.index(finished)]
    todo = [i for i in later if not s.done.get(i)]
    set_task(todo[0] if todo else finished)


def toggle_done(task_id: str) -> None:
    st.session_state.done[task_id] = st.session_state[f"chk_{task_id}"]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title(t("app_title"))
    st.selectbox(
        t("interface_language"), options=list(LOCALES), key="lang",
        format_func=lambda c: LOCALES[c]["language_name"],
    )
    groq_key = st.text_input(t("groq_key"), value=secret("GROQ_API_KEY"), type="password",
                             key="groq_key_in", help=t("groq_key_help"))
    st.text_input(t("youtube_key"), value=secret("YOUTUBE_API_KEY"), type="password",
                  key="yt_key_in", help=t("youtube_key_help"))

    models: list[dict] = []
    if groq_key:
        try:
            models = list_chat_models(groq_key)
        except Exception as e:
            st.error(t("groq_error", err=type(e).__name__))
        if not models:
            st.warning(t("no_models"))
    model_options = [m["id"] for m in models]
    if models and st.session_state.get("model_choice") not in model_options:
        st.session_state.model_choice = pick_default_model(models)
    model_choice = st.selectbox(
        t("model"), options=model_options, key="model_choice",
        help=t("model_help"),
    )
    st.button(t("clear_chat"), on_click=clear_chat)

yt_key = st.session_state.get("yt_key_in") or None
st.markdown(LAYOUT_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Search bar (always on top, like YouTube)
# --------------------------------------------------------------------------- #
bar_search, bar_back = st.columns([10, 1], vertical_alignment="center")
with bar_search:
    with st.form("search_form", border=False, clear_on_submit=False):
        c_input, c_btn = st.columns([8, 1])
        query_in = c_input.text_input(
            t("search_placeholder"), placeholder=t("search_placeholder"), label_visibility="collapsed"
        )
        submitted = c_btn.form_submit_button(t("search_button"))

if st.session_state.view == "watch" and st.session_state.results:
    bar_back.button(t("results_title"), on_click=show_results)

if submitted and query_in.strip():
    vid_from_link = extract_video_id(query_in)
    if vid_from_link:
        select_video(vid_from_link)
    else:
        with st.spinner(t("search_button")):
            do_search(query_in.strip())
        st.session_state.view = "results"

# --------------------------------------------------------------------------- #
# Load the selected video (metadata + transcript)
# --------------------------------------------------------------------------- #
sel = st.session_state.selected_id
if sel and (st.session_state.video is None or st.session_state.video["id"] != sel):
    st.session_state.load_error = None
    with st.spinner(t("loading_video")):
        try:
            chunks = build_chunks(get_transcript(sel))
            if not chunks:
                raise ValueError("EmptyTranscript")
            meta = get_metadata(sel, yt_key)
            hint = next((r for r in st.session_state.results if r["id"] == sel), {})
            meta["title"] = meta.get("title") or hint.get("title", "")
            meta["channel"] = meta.get("channel") or hint.get("channel", "")
            st.session_state.video = {"id": sel, "meta": meta, "chunks": chunks}
            st.session_state.messages = []
            st.session_state.start = 0
            st.session_state.autoplay = False
        except Exception as e:
            st.session_state.video = None
            st.session_state.load_error = t("no_transcript", err=type(e).__name__)
            st.session_state.selected_id = None
            st.session_state.view = "results"


# --------------------------------------------------------------------------- #
# Results grid (a normal scrolling page: videos are the one thing allowed to scroll)
# --------------------------------------------------------------------------- #
def render_footer(pinned: bool) -> None:
    """Footer with the author's name. Pinned to the bottom of the screen on the watch page (no layout impact),
    in the normal page flow elsewhere."""
    css = "app-footer pinned" if pinned else "app-footer"
    st.markdown(f"<div class='{css}'>{t('footer', author=AUTHOR_NAME)}</div>", unsafe_allow_html=True)


def render_results(n_cols: int, prefix: str) -> None:
    if st.session_state.search_error:
        st.error(t("search_failed"))
        return
    results = st.session_state.results
    if not results:
        if st.session_state.query:
            st.info(t("no_results"))
        return
    st.subheader(t("results_title"))
    for i in range(0, len(results), n_cols):
        cols = st.columns(n_cols)
        for col, r in zip(cols, results[i:i + n_cols]):
            with col:
                st.image(r["thumb"])
                st.markdown(f"**{escape_md(truncate(r['title'], 70))}**")
                bits = [
                    r["channel"],
                    t("views", n=fmt_count(r["views"])) if r.get("views") else "",
                    fmt_time(r["duration"]) if r.get("duration") else "",
                ]
                st.caption(" · ".join(b for b in bits if b))
                st.button(t("watch"), key=f"{prefix}_{r['id']}", on_click=select_video, args=(r["id"],))
    if st.session_state.next_token:
        st.button(t("load_more"), key=f"{prefix}_more", on_click=load_more)


# --------------------------------------------------------------------------- #
# Focus panel: to-do list + timer
# --------------------------------------------------------------------------- #
def task_label(task_id: str, minutes: int) -> str:
    return t("task_label", name=t(f"task_{task_id}"), duration=t("minutes", n=minutes))


def render_clock() -> None:
    """Runs as a fragment that re-executes every second while the timer is running."""
    s = st.session_state
    left = timer_remaining()
    mm, ss = divmod(left, 60)
    st.markdown(
        f"<div class='focus-clock'>{t('timer_label', task=t('task_' + s.timer_task), time=f'{mm:02d}:{ss:02d}')}</div>",
        unsafe_allow_html=True,
    )
    if s.timer_running and left <= 0:
        finish_task()
        st.rerun()


def render_focus() -> None:
    s = st.session_state
    if s.pending_done:  # set by finish_task(); applied before the checkboxes are created
        s.done[s.pending_done] = True
        s[f"chk_{s.pending_done}"] = True
        s.pending_done = None
    if s.just_finished:
        st.toast(t("focus_done", task=t("task_" + s.just_finished)))
        s.just_finished = None

    done = sum(1 for i in FOCUS_IDS if s.done.get(i))
    top = st.columns([3, 4, 1, 1, 1], vertical_alignment="center")
    top[0].markdown(f"**{t('focus_title', done=done, total=len(FOCUS_IDS))}**")
    with top[1]:
        st.fragment(run_every=1 if s.timer_running else None)(render_clock)()
    top[2].button("⏸" if s.timer_running else "▶", key="timer_toggle", on_click=timer_toggle,
                  help=t("timer_pause_help") if s.timer_running else t("timer_start_help"))
    top[3].button("↺", key="timer_reset", on_click=timer_reset, help=t("timer_reset_help"))
    top[4].button("⏭", key="timer_next", on_click=timer_next, help=t("timer_next_help"))

    for col, (tid, mins) in zip(st.columns(len(FOCUS_TASKS)), FOCUS_TASKS):
        label = task_label(tid, mins)
        if tid == s.timer_task:
            label = f":blue[**{label}**]"
        # restore the saved value only when Streamlit dropped the widget state (e.g. after the results page)
        extra = {} if f"chk_{tid}" in s else {"value": bool(s.done.get(tid))}
        col.checkbox(label, key=f"chk_{tid}", on_change=toggle_done, args=(tid,), **extra)


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
video = st.session_state.video

if st.session_state.load_error:
    st.error(st.session_state.load_error)

if st.session_state.view != "watch" or not video:
    if not st.session_state.results and not st.session_state.load_error:
        st.info(t("welcome"))
    render_results(4, "home")
    render_footer(pinned=False)
    st.stop()

meta = video["meta"]
render_footer(pinned=True)
left, right = st.columns([3, 2], gap="medium")

# ---- Left: the video is the focus, focus panel right under it ------------- #
with left:
    with st.container(key="video_box"):
        st.video(
            f"https://www.youtube.com/watch?v={video['id']}",
            start_time=int(st.session_state.start),
            autoplay=bool(st.session_state.autoplay),
        )
    info = [
        meta.get("channel"),
        t("views", n=fmt_count(meta["views"])) if meta.get("views") else "",
        meta.get("published"),
    ]
    st.markdown(
        f"<div class='vid-title'>{escape_md(truncate(meta.get('title', ''), 90))}</div>",
        unsafe_allow_html=True,
    )
    st.caption(" · ".join(x for x in info if x))
    with st.container(border=True, key="focus_panel"):
        render_focus()

# ---- Right: AI chat next to the video ------------------------------------- #
STAMP_RE = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")


def render_message(idx: int, m: dict) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            stamps = list(dict.fromkeys(STAMP_RE.findall(m["content"])))[:6]
            if stamps:
                cols = st.columns(len(stamps))
                for col, s_ in zip(cols, stamps):
                    col.button(f"▶ {s_}", key=f"ts_{idx}_{s_}", on_click=jump_to, args=(stamp_to_seconds(s_),))
            if m.get("model"):
                st.caption(m["model"])


with right:
    with st.container(border=True, key="chat_panel"):
        st.markdown(f"<div class='pane-title'>💬 {t('chat_title')}</div>", unsafe_allow_html=True)
        msgs = st.container(height=300, border=False, key="chat_msgs")
        prompt = st.chat_input(t("chat_placeholder"))

    with msgs:
        if not st.session_state.messages and not prompt:
            st.caption(t("chat_empty"))
        for i, m in enumerate(st.session_state.messages):
            render_message(i, m)

        if prompt:
            if not groq_key or not models:
                st.warning(t("need_groq_key") if not groq_key else t("no_models"))
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                context = select_context(video["chunks"], prompt)
                system_prompt = build_system_prompt(meta, context)
                candidates = [model_choice]  # exactly one model, no fallback

                used: dict = {}
                answered = False
                with st.chat_message("assistant"):
                    try:
                        answer = st.write_stream(
                            stream_answer(Groq(api_key=groq_key), candidates, system_prompt,
                                          st.session_state.messages, used)
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "model": used.get("model")}
                        )
                        answered = True
                    except Exception as e:
                        st.error(t("groq_error", err=type(e).__name__))

                if answered:
                    st.rerun()  # re-render so timestamp buttons appear