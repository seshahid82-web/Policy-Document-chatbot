import io
import hashlib

import streamlit as st
from groq import Groq
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

st.set_page_config(page_title="Policy RAG Chatbot", page_icon="📚", layout="wide")

SYSTEM_PROMPT = """You are a Policy Assistant. Answer the user's question using ONLY the policy information provided in the retrieved context.

Rules:
1. Use the retrieved policy documents as the primary source of truth.
2. Do not invent, assume, or add information that is not present in the retrieved context.
3. If multiple policies are relevant, combine their information carefully.
4. Clearly identify the policy document and page when available.
5. If the retrieved information does not answer the question, say exactly: "I could not find sufficient information in the provided policy documents."
6. Give a clear, concise and professional answer.
7. If policies conflict, mention the conflict and identify the relevant sources.
8. Do not use general knowledge to fill missing policy information.
"""

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_pdf_files(service, folder_id):
    query = (
        f"'{folder_id}' in parents and trashed = false "
        "and mimeType = 'application/pdf'"
    )
    response = service.files().list(
        q=query,
        fields="files(id,name,mimeType,modifiedTime)",
        orderBy="name",
        pageSize=100,
    ).execute()
    return response.get("files", [])


def download_pdf(service, file_id):
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def extract_chunks(pdf_bytes, filename):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    chunk_size = 1200
    overlap = 200

    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "text": piece,
                    "source": filename,
                    "page": page_no,
                })
            if end >= len(text):
                break
            start = end - overlap
    return chunks


@st.cache_resource
def build_index(folder_id, drive_version):
    service = get_drive_service()
    files = get_pdf_files(service, folder_id)
    all_chunks = []
    file_names = []

    for file in files:
        data = download_pdf(service, file["id"])
        file_names.append(file["name"])
        all_chunks.extend(extract_chunks(data, file["name"]))

    if not all_chunks:
        raise ValueError("No readable PDF text was found in the Google Drive folder.")

    model = get_embedding_model()
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="policy_documents",
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    for i, c in enumerate(all_chunks):
        ids.append(hashlib.sha256(f"{c['source']}|{c['page']}|{i}".encode()).hexdigest())

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"], "page": c["page"]} for c in all_chunks],
    )
    return collection, file_names, len(all_chunks)


def retrieve(collection, question, top_k=5):
    model = get_embedding_model()
    query_embedding = model.encode([question], normalize_embeddings=True).tolist()[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return result


def ask_groq(question, context):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    prompt = f"""User Question:\n{question}\n\nRetrieved Policy Context:\n{context}\n\nAnswer the question using only the retrieved context. Include source document and page references where available."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


st.title("📚 Policy RAG Chatbot")
st.caption("Ask questions about your private policy PDFs stored in Google Drive.")

required = ["GROQ_API_KEY", "GOOGLE_DRIVE_FOLDER_ID", "gcp_service_account"]
missing = [key for key in required if key not in st.secrets]
if missing:
    st.error("Missing Streamlit secrets: " + ", ".join(missing))
    st.info("Add the required secrets before running the chatbot. See the setup instructions below.")
    st.stop()

folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]

try:
    service = get_drive_service()
    pdf_files = get_pdf_files(service, folder_id)
    drive_version = "|".join(f"{f['id']}:{f.get('modifiedTime','')}" for f in pdf_files)
    collection, names, count = build_index(folder_id, drive_version)
except Exception as exc:
    st.error(f"Could not load policy documents: {exc}")
    st.stop()

with st.sidebar:
    st.header("Policy Documents")
    st.write(f"**PDF files:** {len(names)}")
    st.write(f"**Indexed chunks:** {count}")
    for name in names:
        st.write("• " + name)
    st.divider()
    st.caption("Only PDF files in the configured private Drive folder are indexed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the policies...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the policy documents..."):
            results = retrieve(collection, question, top_k=5)
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            context_parts = []
            for doc, meta, distance in zip(docs, metas, distances):
                context_parts.append(
                    f"Source: {meta.get('source')} | Page: {meta.get('page')} | Similarity distance: {distance:.4f}\n{doc}"
                )
            context = "\n\n---\n\n".join(context_parts)

            if not context:
                answer = "I could not find sufficient information in the provided policy documents."
            else:
                answer = ask_groq(question, context)

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
