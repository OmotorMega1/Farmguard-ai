#!/bin/sh

# Start FastAPI in the background
echo "Starting FastAPI backend on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground
echo "Starting Streamlit dashboard on port 8501..."
streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0