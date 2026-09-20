import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any
import uvicorn
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from router import KuiperRouter
from dynamic_alpha import get_alpha
from llm_client import INBUILT_GROQ_KEY, INBUILT_GEMINI_KEY

# ── APP ────────────────────────────────────────────────────────
app = FastAPI(title="Kuiper Intelligent Routing API", version="2.0.0")

try:
    router = KuiperRouter()
except Exception as e:
    print(f"⚠️ Warning initializing router on boot: {e}")
    import traceback
    traceback.print_exc()
    router = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── TRACES ─────────────────────────────────────────────────────
if os.environ.get("VERCEL"):
    TRACES_FILE = "/tmp/traces.jsonl"
else:
    TRACES_FILE = os.environ.get("TRACES_FILE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces.jsonl")

def save_trace(query, result, latency_ms):
    trace = {
        "query":            query,
        "answer":           result.get("answer"),
        "handled_by":       result.get("handled_by"),
        "layer":            result.get("layer"),
        "layer_name":       result.get("layer_name"),
        "cost_saved":       result.get("cost_saved"),
        "latency_ms":       latency_ms,
        "alpha":            result.get("alpha"),
        "price_tier":       result.get("price_tier"),
        "model":            result.get("model"),
        "selection_reason": result.get("selection_reason"),
        "steps":            result.get("steps", []),
        "timestamp":        datetime.now().isoformat(),
    }
    try:
        with open(TRACES_FILE, "a") as f:
            f.write(json.dumps(trace) + "\n")
    except Exception as e:
        print(f"Error writing trace: {e}")

def load_traces():
    if not os.path.exists(TRACES_FILE):
        return []
    try:
        with open(TRACES_FILE, "r") as f:
            return [json.loads(l) for l in f if l.strip()]
    except Exception:
        return []

# ── MODELS ─────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:    str
    provider: Optional[str] = "auto"
    api_key:  Optional[str] = None

class QueryResponse(BaseModel):
    query:            str
    answer:           str
    handled_by:       str
    layer:            str
    layer_name:       str
    latency_ms:       float
    alpha:            float
    price_tier:       str
    cost_saved:       bool
    confidence:       Optional[float] = 1.0
    model:            Optional[str] = None
    provider:         Optional[str] = None
    selection_reason: Optional[str] = None
    tokens:           Optional[int] = None
    steps:            Optional[List[Any]] = []

# ── STATS TRACKING ─────────────────────────────────────────────
stats = {
    "total": 0,
    "local": 0,
    "llm": 0,
    "saved": 0.0,
    "spent": 0.0,
    "layers": {"0A": 0, "0B": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
}
GPT4_BASELINE_COST = 0.0020

# ── ENDPOINTS ──────────────────────────────────────────────────
@app.get("/")
@app.get("/api")
def root():
    return {
        "message": "Kuiper 7-Layer Intelligent Router API is active 🚀",
        "version": "2.0.0",
        "inbuilt_providers": {
            "groq": bool(INBUILT_GROQ_KEY),
            "gemini": bool(INBUILT_GEMINI_KEY)
        }
    }

@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "pipeline": "7-layer",
        "router_ready": router is not None,
        "version": "2.0.0"
    }

@app.get("/config")
@app.get("/api/config")
def get_config():
    return {
        "inbuilt_groq": bool(INBUILT_GROQ_KEY),
        "inbuilt_gemini": bool(INBUILT_GEMINI_KEY),
        "default_provider": "auto",
        "available_providers": [
            {"id": "auto", "name": "Auto (Intelligent Model Selector)", "inbuilt": True},
            {"id": "groq", "name": "Groq (Ultra-Fast GPT OSS 120B)", "inbuilt": True},
            {"id": "gemini", "name": "Google Gemini (Deep Reasoning & Large Context)", "inbuilt": True},
            {"id": "openai", "name": "OpenAI (Custom Key)", "inbuilt": False},
        ]
    }

@app.post("/query", response_model=QueryResponse)
@app.post("/api/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    global router
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if router is None:
        try:
            router = KuiperRouter()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Router initialization error: {e}")

    start = time.time()
    result = router.route(req.query, provider=req.provider or "auto", api_key=req.api_key)
    latency_ms = round((time.time() - start) * 1000, 2)
    
    cost_saved = result.get("cost_saved", True)
    layer = result.get("layer", "?")

    stats["total"] += 1
    if layer in stats["layers"]:
        stats["layers"][layer] += 1

    if cost_saved:
        stats["local"] += 1
        stats["saved"] += GPT4_BASELINE_COST
    else:
        stats["llm"] += 1
        stats["spent"] += result.get("cost", 0.0001)

    save_trace(req.query, result, latency_ms)

    return QueryResponse(
        query            = req.query,
        answer           = str(result.get("answer", "No answer found")),
        handled_by       = result.get("handled_by", "unknown"),
        layer            = result.get("layer", "?"),
        layer_name       = result.get("layer_name", "Unknown Layer"),
        latency_ms       = latency_ms,
        alpha            = result.get("alpha", 0.5),
        price_tier       = result.get("price_tier", "normal"),
        cost_saved       = cost_saved,
        confidence       = result.get("confidence", 1.0),
        model            = result.get("model"),
        provider         = result.get("provider"),
        selection_reason = result.get("selection_reason"),
        tokens           = result.get("tokens"),
        steps            = result.get("steps", [])
    )

@app.get("/stats")
@app.get("/api/stats")
def get_stats():
    total = stats["total"] or 1
    alpha_info = get_alpha()
    return {
        "total_queries":    stats["total"],
        "local_handled":    stats["local"],
        "llm_calls":        stats["llm"],
        "local_rate":       f"{round((stats['local']/total)*100, 1)}%",
        "estimated_saved":  f"${round(stats['saved'], 4)}",
        "estimated_spent":  f"${round(stats['spent'], 5)}",
        "alpha":            alpha_info["alpha"],
        "price_tier":       alpha_info["tier"],
        "price_tier_name":  alpha_info["tier_name"],
        "tier_color":       alpha_info["color"],
        "price_reasoning":  alpha_info["reasoning"],
        "layer_counts":     stats["layers"]
    }

@app.get("/traces")
@app.get("/api/traces")
def get_traces(limit: int = 30):
    traces = load_traces()
    return {"traces": traces[-limit:], "total": len(traces)}

@app.delete("/traces")
@app.delete("/api/traces")
def clear_traces():
    if os.path.exists(TRACES_FILE):
        try:
            os.remove(TRACES_FILE)
        except Exception:
            pass
    return {"message": "Traces cleared"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)