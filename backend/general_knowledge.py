import re
import threading
import wikipediaapi
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

# ── SETUP ──────────────────────────────────────────────────────
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='Kuiper-App/1.0'
)

DOMAIN_KEYWORDS = [
    "my package", "my order", "my subscription", "my account",
    "my payment", "my card", "my address", "my delivery",
    "refund", "cancel", "charged", "billed", "return",
    "track", "shipping", "dispatch", "invoice",
    # Technical & Software Engineering
    "code structure", "skeleton", "boilerplate", "syntax",
    "c++", "cpp", "python", "java", "rust", "golang", "html",
    "react", "state", "re-render", "closure", "lambda", "settimeout",
    "sql", "injection", "n+1", "big o", "time complexity", "p vs np",
    "isvalidbst", "binary search tree", "rate limiter", "redis",
    "self-attention", "transformer", "lua", "database query", "cors",
    "fastapi", "express", "orm", "algorithm"
]

GK_TRIGGERS = [
    "what is", "what are", "who is", "who was", "who are",
    "tell me about", "explain", "define", "when was", "when is",
    "where is", "capital of", "founder of", "invented by",
    "who created", "who built", "who invented", "who wrote", "who directed",
    "what does", "history of", "meaning of", "tell me who is",
    "who played", "what was", "summary of"
]

STRIP_PHRASES = [
    "tell me about", "tell me who is", "tell me what is",
    "what do you know about", "what is the capital of",
    "what is the meaning of", "what is the history of",
    "what is", "what are", "who is", "who was", "who are",
    "explain", "define", "what was", "where is", "when was", "when is",
    "give me info on", "information about", "what does", "capital of",
]

# ── THREAD-BASED TIMEOUT ───────────────────────────────────────
def run_with_timeout(fn, args=(), timeout=6):
    """
    Run fn(*args) in a thread with a timeout.
    Returns result or None if timeout exceeded.
    O(1) overhead.
    """
    result = [None]
    def target():
        try:
            result[0] = fn(*args)
        except Exception:
            pass
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result[0]

# ── QUERY CLEANER ──────────────────────────────────────────────
def clean_query(text):
    t = text.lower().strip().rstrip('?.!')
    for phrase in STRIP_PHRASES:
        if t.startswith(phrase + " "):
            t = t[len(phrase):].strip()
            break
        elif t.startswith(phrase):
            t = t[len(phrase):].strip()
            break
    return t

# ── CLASSIFIER ─────────────────────────────────────────────────
def is_general_knowledge(text):
    t = text.lower().strip().rstrip('?.!')
    if any(kw in t for kw in DOMAIN_KEYWORDS):
        return False
    # Explicit GK trigger phrase match
    if any(t.startswith(trigger + " ") or t == trigger or trigger in t for trigger in GK_TRIGGERS):
        return True
    # Standalone entity check (1-5 words with no math/operator symbols, e.g. "emma watson", "albert einstein")
    words = t.split()
    if 1 <= len(words) <= 5 and not re.search(r'[\d\+\-\*\/\^\%\(\)\=]', t):
        return True
    return False

# ── WIKIPEDIA ──────────────────────────────────────────────────
def _fetch_wikipedia(query):
    topic = clean_query(query)
    if not topic:
        return None
    page = wiki.page(topic)
    if page.exists():
        raw_summary = page.summary.strip()
        if not raw_summary:
            return None
            
        # Check for disambiguation page
        if "may refer to:" in raw_summary[:80].lower():
            lines = [l.strip() for l in raw_summary.split('\n') if l.strip() and not l.lower().startswith(topic) and "refer to:" not in l.lower()]
            if lines:
                return f"**{page.title}**\n\n" + "\n".join(lines[:3])
            return None

        sentences = raw_summary.split('. ')
        summary = '. '.join(sentences[:2]).strip()
        if not summary.endswith('.'):
            summary += '.'
        return summary
    return None

# ── DUCKDUCKGO ─────────────────────────────────────────────────
def _fetch_duckduckgo(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=1))
        if results:
            return results[0].get('body', None)
    return None

# ── MAIN HANDLER ───────────────────────────────────────────────
def handle(text):
    if not is_general_knowledge(text):
        return {"answer": None, "handled_by": None, "pass_on": True}

    # Try Wikipedia with 6s timeout
    answer = run_with_timeout(_fetch_wikipedia, args=(text,), timeout=6)
    if answer:
        return {
            "answer":     answer,
            "handled_by": "wikipedia",
            "source":     "Wikipedia",
            "pass_on":    False
        }

    # Try DuckDuckGo with 6s timeout
    answer = run_with_timeout(_fetch_duckduckgo, args=(text,), timeout=6)
    if answer:
        return {
            "answer":     answer,
            "handled_by": "duckduckgo",
            "source":     "DuckDuckGo",
            "pass_on":    False
        }

    # Both failed — pass on
    return {"answer": None, "handled_by": None, "pass_on": True}


# ── TEST ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    test_queries = [
        "what is the meaning of life",
        "who is Elon Musk",
        "what is machine learning",
        "what is photosynthesis",
        "where is my package",
        "cancel my subscription",
    ]

    print("🌐 General Knowledge Handler:\n")
    for query in test_queries:
        print(f"  Query: '{query}'")
        start  = time.time()
        result = handle(query)
        ms     = round((time.time()-start)*1000, 1)
        if result["pass_on"]:
            print(f"  ➡️  PASS ON ({ms}ms)\n")
        else:
            print(f"  ✅ [{result['handled_by']}] {result['answer'][:100]} ({ms}ms)\n")