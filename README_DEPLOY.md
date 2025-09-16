# Deployment Quickstart (v5)

This kit includes ready-to-use config for popular hosts.

## Streamlit Community Cloud (easiest)
1) Push your app folder (with `app.py`, `requirements.txt`, `data/`) to GitHub.
2) Go to share.streamlit.io → New app → pick your repo/branch and file path `app.py`.
3) (Optional) Add `runtime.txt` set to `3.11` to pin Python.
4) Deploy.

## Render.com
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Docker (any cloud/VPC)
```
docker build -t hcp-demo:v5 .
docker run -p 8501:8501 hcp-demo:v5
```
Push to any container host (AWS App Runner, Azure Web App for Containers, Google Cloud Run).

## Google Cloud Run (example, Mumbai region)
```
gcloud builds submit --tag gcr.io/PROJECT_ID/hcp-demo:v5
gcloud run deploy hcp-demo --image gcr.io/PROJECT_ID/hcp-demo:v5 --region asia-south1 --platform managed --allow-unauthenticated --port 8501
```

## Heroku / Render Procfile
A `Procfile` is included:
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
