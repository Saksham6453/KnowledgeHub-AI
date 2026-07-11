import re
import tempfile
import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from langchain_nvidia_ai_endpoints import (
    NVIDIAEmbeddings,
    ChatNVIDIA
)

from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# DOCUMENT LOADER (PDF / DOCX / TXT / WEBSITE)

def load_documents(
    uploaded_pdf=None,
    uploaded_docx=None,
    uploaded_txt=None,
    website_url=None
):
    """Loads non-video sources and tags each with 'type' metadata
    so the UI can show what was ingested and where an answer came from."""

    documents = []
    errors = []

    #  PDF
    if uploaded_pdf:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_pdf.getbuffer())
                temp_pdf_path = temp_pdf.name

            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
            for d in docs:
                d.metadata["type"] = "pdf"
                d.metadata["filename"] = uploaded_pdf.name
            documents.extend(docs)
        except Exception as e:
            errors.append(f"PDF loading failed: {e}")

    # DOCX
    if uploaded_docx:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_docx:
                temp_docx.write(uploaded_docx.getbuffer())
                temp_docx_path = temp_docx.name

            loader = Docx2txtLoader(temp_docx_path)
            docs = loader.load()
            for d in docs:
                d.metadata["type"] = "docx"
                d.metadata["filename"] = uploaded_docx.name
            documents.extend(docs)
        except Exception as e:
            errors.append(f"DOCX loading failed: {e}")

    #  TXT
    if uploaded_txt:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_txt:
                temp_txt.write(uploaded_txt.getbuffer())
                temp_txt_path = temp_txt.name

            loader = TextLoader(temp_txt_path, autodetect_encoding=True)
            docs = loader.load()
            for d in docs:
                d.metadata["type"] = "txt"
                d.metadata["filename"] = uploaded_txt.name
            documents.extend(docs)
        except Exception as e:
            errors.append(f"TXT loading failed: {e}")

    # WEBSITE 
    if website_url:
        try:
            url = website_url.strip()
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            loader = WebBaseLoader(url)
            docs = loader.load()
            for d in docs:
                d.metadata["type"] = "website"
                d.metadata["url"] = url
            documents.extend(docs)
        except Exception as e:
            errors.append(f"Website loading failed: {e}")

    print(f"✅ Documents Loaded : {len(documents)}")
    if errors:
        print("⚠️ Loader errors:", errors)

    return documents, errors


# YOUTUBE — TIMESTAMPED TRANSCRIPT LOADER

def extract_video_id(url):
    patterns = [
        r"(?:v=|\/embed\/|\/shorts\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def load_youtube_chunks(youtube_url, chunk_char_limit=900):
    """Fetches the transcript directly (with timestamps) and groups
    consecutive transcript lines into chunks, remembering the start
    time (in seconds) of each chunk so we can later seek the player."""

    video_id = extract_video_id(youtube_url)

    if not video_id:
        raise ValueError("Could not extract a video ID from that YouTube URL.")

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(
            video_id,
            languages=["en", "hi"])
        transcript = transcript.to_raw_data()

    except Exception as e:
        raise ValueError(
            f"Could not fetch a transcript for this video "
            f"(captions may be disabled or unavailable): {e}"
        )

    chunks = []
    current_text = ""
    current_start = None

    for entry in transcript:

        if current_start is None:
            current_start = entry["start"]

        current_text += " " + entry["text"]

        if len(current_text) >= chunk_char_limit:

            chunks.append(
                Document(
                    page_content=current_text.strip(),
                    metadata={
                        "type": "youtube",
                        "video_id": video_id,
                        "url": youtube_url,
                        "start": current_start
                    }
                )
            )

            current_text = ""
            current_start = None

    if current_text.strip():

        chunks.append(
            Document(
                page_content=current_text.strip(),
                metadata={
                    "type": "youtube",
                    "video_id": video_id,
                    "url": youtube_url,
                    "start": current_start or 0
                }
            )
        )

    print(f"✅ YouTube transcript chunked into {len(chunks)} timestamped segments")

    return chunks, video_id

# TEXT SPLITTER

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Total Chunks : {len(chunks)}")
    return chunks


# EMBEDDINGS

def create_embeddings():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is not set. Add it to a .env file "
            "in the project root: NVIDIA_API_KEY=your_key_here"
        )

    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-1b-v2",
        api_key=api_key,
        truncate="END"
    )
    print("✅ Embedding Model Initialized")
    return embeddings

# FAISS VECTOR STORE

def create_vector_store(chunks, embeddings):
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✅ Successfully Stored all Chunks in FAISS")
    return vector_store


def create_retriever(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("✅ Retriever Initialized")
    return retriever

# BUILD PHASE (run once, when user clicks "Process")

def build_knowledge_base(
    uploaded_pdf=None,
    uploaded_docx=None,
    uploaded_txt=None,
    website_url=None,
    youtube_url=None
):
    """
    Loads every source, splits into chunks, embeds, and builds a retriever.
    Runs ONCE per 'Process Knowledge Base' click.

    Returns: (retriever, num_chunks, source_summary, errors)
    """

    documents, errors = load_documents(uploaded_pdf, uploaded_docx, uploaded_txt, website_url)

    # Build small preview snippets per source type for the UI, before chunking
    previews = {}
    for d in documents:
        t = d.metadata.get("type")
        if t and t not in previews:
            previews[t] = d.page_content[:280].strip()

    chunks = split_documents(documents) if documents else []

    youtube_video_id = None
    if youtube_url:
        try:
            yt_chunks, youtube_video_id = load_youtube_chunks(youtube_url)
            chunks.extend(yt_chunks)
        except ValueError as e:
            errors.append(str(e))

    if not chunks:
        raise ValueError(
            "No content could be loaded from the provided sources. "
            + (" | ".join(errors) if errors else "")
        )

    embeddings = create_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    retriever = create_retriever(vector_store)

    source_summary = {
        "pdf": {"name": uploaded_pdf.name, "size": uploaded_pdf.size, "preview": previews.get("pdf", "")}
        if uploaded_pdf else None,
        "docx": {"name": uploaded_docx.name, "size": uploaded_docx.size, "preview": previews.get("docx", "")}
        if uploaded_docx else None,
        "txt": {"name": uploaded_txt.name, "size": uploaded_txt.size, "preview": previews.get("txt", "")}
        if uploaded_txt else None,
        "website": {"url": website_url, "preview": previews.get("website", "")}
        if website_url else None,
        "youtube": {"url": youtube_url, "video_id": youtube_video_id}
        if youtube_url and youtube_video_id else None,
    }

    return retriever, len(chunks), source_summary, errors


# QUERY PHASE (run on every chat message)

def retrieve_documents(retriever, question):
    docs = retriever.invoke(question)
    print(f"✅ Retrieved {len(docs)} documents")
    return docs


def create_context(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    print("✅ Context Created")
    return context

#LLM
def create_LLM():
    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        timeout=120
    )
    return llm

#PROMPT CREATION
def create_prompt():
    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.

Use ONLY the provided context.

If the answer cannot be found in the context, respond exactly:

"I don't know based on the provided context."

Do not make up information.

Context:
{context}

Question:
{question}
""")
    return prompt

#FINAL RESPONSE
def generate_response(llm, prompt, context, question):
    formatted_prompt = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(formatted_prompt)
    print("✅ Response Generated")
    return response.content


def answer_question(retriever, question):
    """
    Runs retrieval + generation. Also detects whether the most relevant
    chunk came from the YouTube video, and if so, returns its timestamp
    so the UI can seek the player there.

    Returns: (answer, docs, video_timestamp_seconds_or_None)
    """
    docs = retrieve_documents(retriever, question)
    context = create_context(docs)
    prompt = create_prompt()
    llm = create_LLM()
    answer = generate_response(llm, prompt, context, question)

    video_timestamp = None
    for d in docs:
        if d.metadata.get("type") == "youtube" and d.metadata.get("start") is not None:
            video_timestamp = d.metadata["start"]
            break  # docs are ordered by relevance; take the top youtube hit

    return answer, docs, video_timestamp