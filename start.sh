#!/bin/bash
set -e

# Activate virtual environment if it exists (optional, for local dev)
# source venv/bin/activate

# Run the application
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
