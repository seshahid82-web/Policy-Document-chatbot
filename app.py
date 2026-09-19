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

SYSTEM_PROMPT = """You are a Policy Assistant.
Answer the user's question using ONLY the retrieved policy context.

Rules:
1. Do not invent, assume, or add policy information that is not in the context.
2. If the answer is not supported by the context, say exactly:
   "I could not find sufficient information in the provided policy documents."
3. If more than one policy is relevant, combine the information carefully.
4. Mention the policy document name and page number when available.
5. Keep the answer clear, concise, and professional.
6. Do not use general knowledge to fill missing policy information.
"""

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_secret(name):
    if name not in st.secrets:
        raise ValueError(f"Missing Streamlit secret: {name}")
    return st.secrets[name]


@st.cache_resource
def get_drive_service():
    service_account_info = dict(st.secrets["gcp_service_account"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def get_pdf_files(service, folder_id):
    query = (
        f"'{folder_id}' in parents and trashed = false "
        "and mimeType = 'application/pdf'"
    )

    files = []
    page_token = None

    while True:
        response = service.files().list(
            q=query,
            fields="nextPageToken, files(id,name,mimeType,modifiedTime)",
            orderBy="name",
            pageSize=100,
            pageToken=page_token,
        ).execute()

        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return files


def download_pdf(service, file_id):
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False

    while not done:
        _, done = downloader.next_chunk()

    return buffer.getvalue()


def split_text(text, chunk_size=1200, overlap=200):
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


def extract_chunks(pdf_bytes, filename):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        for piece in split_text(text):
            chunks.append(
                {
                    "text": piece,
                    "source": filename,
                    "page": page_number,
                }
            )

    return chunks


@st.cache_resource(show_spinner="Reading policy PDFs and creating the search index...")
def build_index(folder_id, drive_version):
    service = get_drive_service()
    pdf_files = get_pdf_files(service, folder_id)

    all_chunks = []
    file_names = []

    for pdf_file in pdf_files:
        data = download_pdf(service, pdf_file["id"])
        file_names.append(pdf_file["name"])
        all_chunks.extend(extract_chunks(data, pdf_file["name"]))

    if not all_chunks:
        raise ValueError(
            "No readable PDF text was found. Check that the Google Drive folder "
            "contains PDF files and that it is shared with the service-account email."
        )

    model = get_embedding_model()
    texts = [item["text"] for item in all_chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    client = chromadb.Client()
    collection_name = "policy_documents"

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    metadatas = []

    for index, item in enumerate(all_chunks):
        unique_id = hashlib.sha256(
            f"{item['source']}|{item['page']}|{index}".encode("utf-8")
        ).hexdigest()
        ids.append(unique_id)
        metadatas.append(
            {
                "source": item["source"],
                "page": item["page"],
            }
        )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection, file_names, len(all_chunks)


def retrieve(collection, question, top_k=5):
    model = get_embedding_model()
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()[0]

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def ask_groq(question, context):
    client = Groq(api_key=get_secret("GROQ_API_KEY"))

    user_prompt = f"""User Question:
{question}

Retrieved Policy Context:
{context}

Answer using only the retrieved policy context. Cite the policy document and page number in your answer where available."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_completion_tokens=1000,
    )

    return response.choices[0].message.content


st.title("📚 Policy RAG Chatbot")
st.caption("Ask questions about your private policy PDFs stored in Google Drive.")

# Check required secrets before doing any Google Drive or model work.
required = ["GROQ_API_KEY", "GOOGLE_DRIVE_FOLDER_ID", "gcp_service_account"]
missing = [name for name in required if name not in st.secrets]

if missing:
    st.error("Missing Streamlit Secrets: " + ", ".join(missing))
    st.markdown(
        """
### Required Secrets

Add these to **Streamlit → App → Settings → Secrets**:

```toml
GROQ_API_KEY = "your-groq-api-key"
GOOGLE_DRIVE_FOLDER_ID = "your-google-drive-folder-id"

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\\nYOUR_KEY\\n-----END PRIVATE KEY-----\\n"
client_email = "your-service-account-email"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-client-certificate-url"
```

**Do not put these secrets in GitHub.**
"""
    )
    st.stop()

folder_id = str(st.secrets["GOOGLE_DRIVE_FOLDER_ID"]).strip()

if not folder_id:
    st.error("GOOGLE_DRIVE_FOLDER_ID is empty.")
    st.stop()

try:
    service = get_drive_service()
    pdf_files = get_pdf_files(service, folder_id)

    if not pdf_files:
        st.error("No PDF files were found in the Google Drive folder.")
        st.info(
            "Make sure the folder contains PDF files and is shared with the "
            "Google service-account email as Viewer."
        )
        st.stop()

    drive_version = "|".join(
        f"{item['id']}:{item.get('modifiedTime', '')}" for item in pdf_files
    )

    collection, names, chunk_count = build_index(folder_id, drive_version)

except Exception as exc:
    st.error("Could not load the policy documents.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("📄 Policy Documents")
    st.write(f"**PDF files:** {len(names)}")
    st.write(f"**Indexed chunks:** {chunk_count}")

    for name in names:
        st.write("• " + name)

    st.divider()
    st.caption("Only PDF files inside the configured private Google Drive folder are indexed.")

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
        with st.spinner("Searching policy documents..."):
            try:
                results = retrieve(collection, question, top_k=5)
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]

                context_parts = []
                for document, metadata in zip(documents, metadatas):
                    context_parts.append(
                        f"Source: {metadata.get('source')} | "
                        f"Page: {metadata.get('page')}\n{document}"
                    )

                context = "\n\n---\n\n".join(context_parts)

                if not context:
                    answer = "I could not find sufficient information in the provided policy documents."
                else:
                    answer = ask_groq(question, context)

            except Exception as exc:
                answer = "An error occurred while processing your question."
                st.exception(exc)

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
