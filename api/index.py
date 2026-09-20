import sys
import os

# Set environment for Vercel serverless
os.environ["VERCEL"] = "1"

# Add backend directory to Python module search path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Import the existing FastAPI app from backend/main.py
from main import app
