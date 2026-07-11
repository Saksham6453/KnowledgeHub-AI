import streamlit as st
from rag import build_knowledge_base, answer_question

# PAGE CONFIG

st.set_page_config(
    page_title="KnowledgeHub AI",
    page_icon="📚",
    layout="wide"
)
# CUSTOM CSS

st.markdown("""
<style>
.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 { margin: 0; font-size: 1.9rem; }
.hero p { margin: 0.35rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

.card {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    background: #f7f8fc;
    border: 1px solid #e6e8f0;
    margin-bottom: 0.7rem;
}
.card-title {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}
.card-sub {
    font-size: 0.78rem;
    color: #6b7280;
}
.badge {
    display: inline-block;
    padding: 0.12rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 0.3rem;
}
.badge-pdf { background: #fde2e2; color: #b42318; }
.badge-docx { background: #dbeafe; color: #1d4ed8; }
.badge-txt { background: #e0f2fe; color: #0369a1; }
.badge-website { background: #dcfce7; color: #15803d; }
.badge-youtube { background: #fee2e2; color: #dc2626; }
.empty-state {
    text-align: center;
    padding: 2rem 1rem;
    color: #9ca3af;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# SESSION STATE INIT

defaults = {
    "retriever": None,
    "num_chunks": 0,
    "source_summary": {},
    "chat_history": [],
    "video_start_time": 0,
    "documents": []
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# HEADER

st.markdown("""
<div class="hero">
    <h1>📚 KnowledgeHub AI</h1>
    <p>Chat with PDFs, DOCX, TXT files, Websites and YouTube Videos — powered by LangChain + NVIDIA Llama + FAISS</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR

with st.sidebar:
    st.header("⚙️ Configuration")
    st.success("LLM")
    st.write("NVIDIA Llama 3.1 (8B Instruct)")
    st.success("Embeddings")
    st.write("NVIDIA Nemotron Embed")
    st.success("Vector Database")
    st.write("FAISS")
    st.success("Framework")
    st.write("LangChain")

    st.divider()
    st.header("📊 Status")
    st.metric("Chunks Indexed", st.session_state["num_chunks"])
    st.metric("Retriever", "Top-3")

    if st.session_state["retriever"] is not None:
        st.divider()
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            for key, val in defaults.items():
                st.session_state[key] = val
            st.rerun()

# UPLOAD SECTION

with st.expander("➕ Add Knowledge Sources", expanded=(st.session_state["retriever"] is None)):

    col1, col2 = st.columns(2)

    with col1:
        uploaded_pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])
        uploaded_docx = st.file_uploader("📄 Upload DOCX", type=["docx"])
        uploaded_txt = st.file_uploader("📝 Upload TXT", type=["txt"])

    with col2:
        website_url = st.text_input("🌐 Website URL")
        youtube_url = st.text_input("🎥 YouTube URL")

    if st.button("🚀 Process Knowledge Base", use_container_width=True, type="primary"):
        if (uploaded_pdf is None and uploaded_docx is None and
                uploaded_txt is None and website_url == "" and youtube_url == ""):
            st.warning("⚠️ Please upload a file or enter a URL.")
        else:
            with st.spinner("Loading sources, chunking, and building embeddings... this can take a minute."):
                try:
                    retriever, num_chunks, source_summary, load_errors = build_knowledge_base(
                        uploaded_pdf=uploaded_pdf,
                        uploaded_docx=uploaded_docx,
                        uploaded_txt=uploaded_txt,
                        website_url=website_url,
                        youtube_url=youtube_url
                    )
                    st.session_state["retriever"] = retriever
                    st.session_state["num_chunks"] = num_chunks
                    st.session_state["source_summary"] = source_summary
                    st.session_state["chat_history"] = []
                    st.session_state["video_start_time"] = 0

                    st.success(f"✅ Knowledge Base Ready! Indexed {num_chunks} chunks.")
                    for err in load_errors:
                        st.warning(f"⚠️ {err}")

                    st.rerun()

                except ValueError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error while building the knowledge base: {e}")

st.divider()


# HELPERS

def format_time(seconds):
    if seconds is None:
        return "0:00"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def human_size(num_bytes):
    if num_bytes is None:
        return ""
    for unit in ["B", "KB", "MB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


# MAIN LAYOUT — CHAT PROCESSED FIRST SO THE
# SOURCE PANEL BELOW REFLECTS THE LATEST STATE


question = st.chat_input("Ask anything about your documents...")

if question:
    st.session_state["chat_history"].append({"role": "user", "content": question})

    if st.session_state["retriever"] is None:
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": "⚠️ Please upload documents/URLs and click 'Process Knowledge Base' first."
        })
    else:
        try:
            answer, docs, video_timestamp = answer_question(
                retriever=st.session_state["retriever"],
                question=question
            )
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": answer,
                "docs": docs
            })
            st.session_state["documents"] = docs

            if video_timestamp is not None:
                st.session_state["video_start_time"] = int(video_timestamp)

        except Exception as e:
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": f"❌ Error generating a response: {e}"
            })

col_sources, col_chat = st.columns([1, 1.3], gap="large")

# SOURCES PANEL

with col_sources:
    st.subheader("🗂️ Knowledge Sources")

    summary = st.session_state["source_summary"]

    if not summary:
        st.markdown('<div class="empty-state">No sources loaded yet.<br>Add sources above and click "Process Knowledge Base".</div>', unsafe_allow_html=True)
    else:
        tab_docs, tab_web, tab_video = st.tabs(["📄 Documents", "🌐 Website", "🎥 Video"])

        # Documents tab
        with tab_docs:
            any_doc = False
            for key, badge_class, icon in [("pdf", "badge-pdf", "📄"), ("docx", "badge-docx", "📄"), ("txt", "badge-txt", "📝")]:
                info = summary.get(key)
                if info:
                    any_doc = True
                    st.markdown(f"""
                    <div class="card">
                        <span class="badge {badge_class}">{key.upper()}</span>
                        <div class="card-title">{icon} {info['name']}</div>
                        <div class="card-sub">{human_size(info.get('size'))}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if info.get("preview"):
                        with st.expander("Preview text"):
                            st.caption(info["preview"] + "...")
            if not any_doc:
                st.markdown('<div class="empty-state">No documents uploaded.</div>', unsafe_allow_html=True)

        # Website tab 
        with tab_web:
            web_info = summary.get("website")
            if web_info:
                st.markdown(f"""
                <div class="card">
                    <span class="badge badge-website">WEBSITE</span>
                    <div class="card-title">🌐 {web_info['url']}</div>
                </div>
                """, unsafe_allow_html=True)
                try:
                    st.components.v1.iframe(web_info["url"], height=400, scrolling=True)
                except Exception:
                    pass
                st.caption("If the preview above is blank, the site blocks embedding — use the link instead.")
                st.link_button("Open website ↗", web_info["url"])
                if web_info.get("preview"):
                    with st.expander("Extracted text preview"):
                        st.caption(web_info["preview"] + "...")
            else:
                st.markdown('<div class="empty-state">No website loaded.</div>', unsafe_allow_html=True)

        # Video tab 
        with tab_video:
            yt_info = summary.get("youtube")
            if yt_info:
                st.markdown(f"""
                <div class="card">
                    <span class="badge badge-youtube">YOUTUBE</span>
                    <div class="card-title">🎥 Video ID: {yt_info['video_id']}</div>
                    <div class="card-sub">Ask a question and the player jumps to the relevant moment.</div>
                </div>
                """, unsafe_allow_html=True)
                st.video(yt_info["url"], start_time=st.session_state["video_start_time"])
                if st.session_state["video_start_time"]:
                    st.caption(f"▶️ Jumped to {format_time(st.session_state['video_start_time'])}")
            else:
                st.markdown('<div class="empty-state">No video loaded.</div>', unsafe_allow_html=True)

# CHAT PANEL 

with col_chat:
    st.subheader("💬 Chat")

    if not st.session_state["chat_history"]:
        st.markdown('<div class="empty-state">Ask a question about your sources to get started.</div>', unsafe_allow_html=True)

    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

            docs = turn.get("docs")
            if docs:
                with st.expander("📚 Retrieved sources"):
                    for i, doc in enumerate(docs, start=1):
                        d_type = doc.metadata.get("type", "unknown")
                        badge_class = f"badge-{d_type}" if d_type in ["pdf", "docx", "txt", "website", "youtube"] else ""
                        label = doc.metadata.get("filename") or doc.metadata.get("url") or d_type

                        st.markdown(f'<span class="badge {badge_class}">{d_type.upper()}</span> **{label}**', unsafe_allow_html=True)

                        if d_type == "youtube":
                            ts = doc.metadata.get("start")
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.caption(doc.page_content[:200] + "...")
                            with col_b:
                                if st.button(f"▶ {format_time(ts)}", key=f"jump_{i}_{turn['content'][:20]}"):
                                    st.session_state["video_start_time"] = int(ts)
                                    st.rerun()
                        else:
                            st.caption(doc.page_content[:200] + "...")

                        st.markdown("---")


# FOOTER

st.divider()
st.caption("Built using Streamlit • LangChain • NVIDIA AI Endpoints • FAISS")