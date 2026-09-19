# Policy RAG Chatbot

A beginner-friendly Streamlit RAG chatbot that reads private PDF policy documents from a Google Drive folder, creates embeddings, retrieves relevant sections, and uses Groq to generate an answer.

## Project files

- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies
- `.gitignore` - prevents secrets from being committed

## Streamlit secrets

Create `.streamlit/secrets.toml` locally, or paste the same contents into Streamlit Community Cloud Secrets.

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
GOOGLE_DRIVE_FOLDER_ID = "YOUR_GOOGLE_DRIVE_FOLDER_ID"

[gcp_service_account]
type = "service_account"
project_id = "YOUR_PROJECT_ID"
private_key_id = "YOUR_PRIVATE_KEY_ID"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END PRIVATE KEY-----\n"
client_email = "YOUR_SERVICE_ACCOUNT_EMAIL"
client_id = "YOUR_CLIENT_ID"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "YOUR_CLIENT_CERTIFICATE_URL"
```

Share the private Google Drive policy folder with the service-account email as Viewer. Put the folder ID (the value after `/folders/` in the Drive URL) in `GOOGLE_DRIVE_FOLDER_ID`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push `app.py`, `requirements.txt`, `.gitignore`, and `README.md` to GitHub. Do **not** push `.streamlit/secrets.toml`. Add the secrets in your Streamlit deployment settings.
