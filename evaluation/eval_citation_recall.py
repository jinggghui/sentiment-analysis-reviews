"""
Automated Citation Recall Evaluation for RAG Pipeline

Generates N diverse queries from templates, runs the full RAG pipeline,
and computes Citation Recall = |cited ∩ retrieved| / |cited|.
"""

import os
import re
import random
import argparse
import sqlite3
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# PATH CONFIG (Colab + Local + GitHub)
# =========================

def get_paths():
    if os.path.exists("/content/drive"):
        base = "/content/drive/MyDrive/cs6120_rag_project"
        return {
            "db_path": os.path.join(base, "database", "reviews.db"),
            "hf_cache": os.path.join(base, "hf_cache"),
        }

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "db_path": os.path.join(base_dir, "data", "reviews.db"),
        "hf_cache": os.path.join(base_dir, "hf_cache"),
    }


PATHS = get_paths()
DB_PATH = PATHS["db_path"]
HF_CACHE_DIR = PATHS["hf_cache"]


# =========================
# CONFIG
# =========================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

TOP_K = 5
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# =========================
# QUERY GENERATION
# =========================

TEMPLATES = [
    "What do users say about {topic}?",
    "Do users complain about {topic}?",
    "What are common issues with {topic}?",
    "How do users feel about {topic}?",
    "What {topic} problems do users report?",
    "Are there complaints about {topic}?",
    "What feedback exists about {topic}?",
    "Do reviews mention {topic}?",
    "What causes {topic} according to users?",
    "Is {topic} a common issue?",
]

TOPICS = [
    "battery drain", "app crashes", "slow loading", "login failures",
    "ads", "notifications", "updates", "freezing", "performance",
    "UI design", "dark mode", "offline mode", "storage usage",
    "privacy", "permissions", "data loss", "sync issues",
    "customer support", "pricing", "subscriptions", "in-app purchases",
    "camera quality", "video playback", "audio issues", "download speed",
    "screen rotation", "keyboard", "search functionality", "filters",
    "sharing features", "account security", "password reset", "two-factor auth",
    "push notifications", "email notifications", "widget support",
    "accessibility", "font size", "language support", "translation quality",
    "map accuracy", "GPS tracking", "location services", "bluetooth",
    "wifi connectivity", "cellular data usage", "background processes",
    "startup time", "memory usage", "heating issues", "compatibility",
    "tablet support", "landscape mode", "split screen", "gesture controls",
    "voice commands", "chatbot", "help section", "tutorials",
    "onboarding", "registration", "profile settings", "backup",
    "export features", "import features", "file management", "photo editing",
    "social features", "messaging", "group chats", "video calls",
    "payment methods", "refund process", "billing errors", "free trial",
    "content quality", "recommendations", "personalization", "user interface",
    "navigation", "menu layout", "button responsiveness", "scrolling",
    "animation lag", "pop-ups", "redirect issues", "error messages",
    "loading screens", "splash screen", "app size", "update frequency",
    "changelog clarity", "feature removal", "feature requests", "beta features",
    "rating prompts", "review responses", "developer communication",
    "community features", "leaderboards", "achievements", "rewards",
    "game mechanics", "difficulty level", "content updates", "server issues",
    "multiplayer", "matchmaking", "lag spikes", "disconnections",
]


def generate_queries(n: int, seed: int = SEED):
    rng = random.Random(seed)
    queries = []

    for topic in TOPICS:
        for template in TEMPLATES:
            queries.append(template.format(topic=topic))

    rng.shuffle(queries)
    return queries[:n]


# =========================
# DATA LOADING
# =========================

def load_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT review_id, review_text, sentiment, app_name FROM reviews")
    rows = cur.fetchall()

    review_ids = [r[0] for r in rows]
    reviews = [r[1] for r in rows]
    sentiments = [r[2] for r in rows]
    app_names = [r[3] for r in rows]

    cur.execute("SELECT embedding FROM reviews ORDER BY review_id")
    embeddings = np.array([
        np.frombuffer(row[0], dtype=np.float32) if row[0] else np.zeros(384)
        for row in cur.fetchall()
    ])

    conn.close()

    return {
        "reviews": reviews,
        "review_ids": review_ids,
        "sentiments": sentiments,
        "app_names": app_names,
        "embeddings": embeddings,
    }


# =========================
# RETRIEVAL
# =========================

def build_bm25(reviews):
    tokenized = [r.lower().split() for r in reviews]
    return BM25Okapi(tokenized), tokenized


def hybrid_search(query, bm25, tokenized, embeddings, embed_model, top_k=TOP_K):
    tokens = query.lower().split()

    bm25_scores = bm25.get_scores(tokens)
    bm25_idx = np.argsort(bm25_scores)[::-1][:top_k * 2]
    bm25_norm = bm25_scores[bm25_idx] / (bm25_scores[bm25_idx].max() + 1e-10)

    qvec = embed_model.encode(query, convert_to_numpy=True)
    sims = np.dot(embeddings, qvec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(qvec) + 1e-10
    )

    vec_idx = np.argsort(sims)[::-1][:top_k * 2]
    vec_norm = sims[vec_idx] / (sims[vec_idx].max() + 1e-10)

    combined = {}

    for i, s in zip(bm25_idx, bm25_norm):
        combined[i] = BM25_WEIGHT * s

    for i, s in zip(vec_idx, vec_norm):
        combined[i] = combined.get(i, 0) + VECTOR_WEIGHT * s

    top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [i for i, _ in top]


# =========================
# PROMPT + GENERATION
# =========================

def create_prompt(query, reviews, review_ids, app_names, sentiments):
    ctx = ""
    for rid, text, app, sent in zip(review_ids, reviews, app_names, sentiments):
        ctx += f"[Review #{rid}] ({app}) ({sent})\n{text}\n\n"

    return f"""
You are a helpful assistant analyzing app reviews.

Based on these user reviews:
{ctx}

Answer the question: {query}

Guidelines:
1. Use the reviews above
2. Cite using [Review #ID]
3. Do NOT hallucinate
4. Be concise

Answer:
"""


def generate_response(prompt, tokenizer, model, max_new_tokens=256):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("Answer:")[-1].strip()


def extract_citations(response):
    return [int(x) for x in re.findall(r"\[Review #(\d+)\]", response)]


# =========================
# EVALUATION
# =========================

def evaluate_citation_recall(n_queries=1000):
    print(f"[INFO] Using DB_PATH: {DB_PATH}")
    print(f"[INFO] Using HF_CACHE_DIR: {HF_CACHE_DIR}")

    print("Loading data...")
    data = load_database()

    print("Building BM25...")
    bm25, tokenized = build_bm25(data["reviews"])

    print("Loading models...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=HF_CACHE_DIR,
    )

    queries = generate_queries(n_queries)

    total_cited = 0
    grounded = 0
    no_citation = 0

    for query in tqdm(queries, desc="Evaluating"):
        idx = hybrid_search(query, bm25, tokenized, data["embeddings"], embed_model)

        retrieved_ids = [data["review_ids"][i] for i in idx]
        retrieved_reviews = [data["reviews"][i] for i in idx]
        retrieved_apps = [data["app_names"][i] for i in idx]
        retrieved_sent = [data["sentiments"][i] for i in idx]

        prompt = create_prompt(query, retrieved_reviews, retrieved_ids, retrieved_apps, retrieved_sent)
        response = generate_response(prompt, tokenizer, model)

        cited = extract_citations(response)

        if len(cited) == 0:
            no_citation += 1
            continue

        for cid in cited:
            total_cited += 1
            if cid in retrieved_ids:
                grounded += 1

    recall = grounded / total_cited if total_cited > 0 else 0

    print("\n==== RESULTS ====")
    print(f"Queries: {len(queries)}")
    print(f"No citation: {no_citation}")
    print(f"Total cited: {total_cited}")
    print(f"Grounded: {grounded}")
    print(f"Citation Recall: {recall:.4f}")

    return recall


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_queries", type=int, default=1000)
    args = parser.parse_args()

    evaluate_citation_recall(args.n_queries)