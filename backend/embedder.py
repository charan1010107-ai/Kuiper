from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

# ── MODEL ──────────────────────────────────────────────────────
# Small, fast, runs locally — no API call needed
MODEL_NAME    = 'all-MiniLM-L6-v2'
CACHE_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embedding_cache.json')

class Embedder:
    def __init__(self):
        print("⚙️  Loading embedding model...")
        self.model  = SentenceTransformer(MODEL_NAME)
        self.cache  = self._load_cache()
        print(f"✅ Embedder ready. Cache has {len(self.cache)} entries.\n")

    # ── PERSISTENT CACHE ───────────────────────────────────────
    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(CACHE_FILE, 'w') as f:
            json.dump(self.cache, f)

    # ── EMBED ──────────────────────────────────────────────────
    def embed(self, text):
        """Convert text → 384-dim vector. Cache result for reuse."""
        key = text.lower().strip()

        if key in self.cache:
            print(f"  💾 Embedding cache hit: '{text}'")
            return np.array(self.cache[key])

        # Not in cache — compute it
        vector = self.model.encode([text])[0]
        self.cache[key] = vector.tolist()
        self._save_cache()
        print(f"  🔢 Computed new embedding: '{text}'")
        return vector

    # ── SIMILARITY ─────────────────────────────────────────────
    def similarity(self, vec1, vec2):
        """Cosine similarity between two vectors (0 to 1)"""
        dot    = np.dot(vec1, vec2)
        norms  = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(dot / norms) if norms > 0 else 0.0

    # ── FIND MOST SIMILAR ──────────────────────────────────────
    def find_similar(self, query_vec, labeled_embeddings, top_k=1):
        """
        Given a query vector, find most similar from a labeled set.
        labeled_embeddings = list of {"text": ..., "label": ..., "vector": ...}
        """
        if not labeled_embeddings:
            return None

        scores = []
        for item in labeled_embeddings:
            sim = self.similarity(query_vec, item["vector"])
            scores.append((sim, item))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_sim, top_item = scores[0]

        return {
            "matched_text": top_item["text"],
            "label":        top_item["label"],
            "similarity":   round(top_sim, 3)
        }


# ── TEST ───────────────────────────────────────────────────────
if __name__ == "__main__":
    embedder = Embedder()

    # Simulate labeled embeddings from past LLM traces
    print("📚 Building labeled embedding store from traces...\n")
    trace_texts = [
        ("where is my package",       "shipping_query"),
        ("i was billed two times",    "payment_issue"),
        ("cancel my subscription",    "cancellation"),
        ("give me my money back",     "refund_request"),
        ("update my delivery address","address_update"),
        ("my payment failed",         "payment_issue"),
        ("i want to return this item","refund_request"),
    ]

    labeled_embeddings = []
    for text, label in trace_texts:
        vec = embedder.embed(text)
        labeled_embeddings.append({
            "text":   text,
            "label":  label,
            "vector": vec
        })

    # Now test with queries that MISSED tinyml and cache
    print("\n🔍 Finding similar traces for new queries:\n")
    test_queries = [
        "i was charged twice",           # similar to payment_issue
        "refund my money",               # similar to refund_request
        "the quantum flux is unstable",  # no match → low similarity
        "stop my plan",                  # similar to cancellation
        "has my parcel arrived yet",     # similar to shipping_query
    ]

    SIMILARITY_THRESHOLD = 0.5  # minimum to trust the match

    for query in test_queries:
        print(f"Query: '{query}'")
        query_vec = embedder.embed(query)
        result    = embedder.find_similar(query_vec, labeled_embeddings)

        if result and result["similarity"] >= SIMILARITY_THRESHOLD:
            print(f"  ✅ MATCHED → {result['label']}")
            print(f"     Closest trace: '{result['matched_text']}'")
            print(f"     Similarity: {result['similarity']}\n")
        else:
            sim = result["similarity"] if result else 0
            print(f"  ❓ NO CONFIDENT MATCH (similarity: {sim})")
            print(f"     → Pass to Surrogate Ensemble\n")