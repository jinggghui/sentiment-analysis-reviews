import streamlit as st
import pickle
import numpy as np
import re
import requests
from pathlib import Path

st.set_page_config(page_title="App Review RAG", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTitle {
        font-size: 3rem !important;
        color: #1f77b4;
        font-weight: bold;
    }
    .stMetric {
        background-color: #666666;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all machine learning models just once and cache them in memory."""
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    
    # Load embedding model
    embedding_model = SentenceTransformer('./saved_models/embedding_model', trust_remote_code=True)
    
    # Load BM25
    with open('./saved_models/bm25.pkl', 'rb') as f:
        bm25 = pickle.load(f)
    
    # Load review data
    with open('./saved_models/review_data.pkl', 'rb') as f:
        review_data = pickle.load(f)
        
    return embedding_model, bm25, review_data

# Header
st.title("📱 App Review RAG System")
st.markdown("*Intelligent search & analysis of app reviews with citations*")
st.markdown("---")

# Loading section
with st.spinner("Loading models into memory (only happens once)..."):
    try:
        embedding_model, bm25, review_data = load_models()
        
        review_ids = review_data['review_ids']
        review_texts = review_data['review_texts']
        review_ratings = review_data['review_ratings']
        review_dates = review_data['review_dates']
        review_embeddings = review_data['review_embeddings']
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

# ============================================================================
# RETRIEVAL FUNCTION
# ============================================================================
def retrieve_reviews(query, top_k=5):
    """Retrieve top-k reviews using BM25 + vector similarity"""
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    query_norm = np.linalg.norm(query_embedding)
    reviews_norm = np.linalg.norm(review_embeddings, axis=1)
    
    dot_product = np.dot(review_embeddings, query_embedding)
    vector_scores = dot_product / (query_norm * reviews_norm)
    vector_scores = (vector_scores + 1) / 2
    
    max_bm25 = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1
    bm25_normalized = bm25_scores / max_bm25
    
    combined_scores = 0.4 * bm25_normalized + 0.6 * vector_scores
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    return top_indices

# ============================================================================
# ANSWER GENERATION FUNCTION
# ============================================================================
def generate_answer_with_citations(query, top_reviews, max_tokens=256):
    """Generate answer using LLM with citations"""
    reviews_text = ""
    for review in top_reviews:
        reviews_text += f"[Review #{review['id']}] (Rating: {review['rating']}⭐): {review['text']}\n\n"
    
    system_prompt = """You are a helpful assistant analyzing app reviews.
Your job is to answer questions about reviews using ONLY the retrieved reviews below.

IMPORTANT RULES:
1. Always cite reviews using format [Review #ID]
2. Never make up information not in the reviews
3. Be concise (2-3 sentences max)
4. If you don't know, say "I don't have enough information"

Retrieved Reviews:
""" + reviews_text
    
    user_message = f"Question: {query}\n\nAnswer (cite reviews):"
    prompt = system_prompt + user_message
    
    # Try different local hostnames (in case running in Docker)
    ollama_urls = [
        "http://host.docker.internal:11434/api/generate",
        "http://localhost:11434/api/generate",
        "http://127.0.0.1:11434/api/generate"
    ]
    
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": max_tokens
        }
    }
    
    for url in ollama_urls:
        try:
            response = requests.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                answer = response.json().get("response", "").strip()
                answer = answer.split("Answer (cite reviews):")[-1].strip()
                return answer
        except requests.exceptions.RequestException:
            continue
            
    return "Error: Could not connect to Ollama. Please ensure you have Ollama installed and have run 'ollama run mistral' in your terminal."

# ============================================================================
# CACHED TOPIC GENERATION
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def generate_macro_topic_summary(topic_query):
    top_indices = retrieve_reviews(topic_query, top_k=5)
    top_reviews = []
    reviews_text = ""
    for idx in top_indices:
        r_dict = {
            'id': review_ids[idx],
            'rating': review_ratings[idx],
            'date': review_dates[idx],
            'text': review_texts[idx]
        }
        top_reviews.append(r_dict)
        reviews_text += f"- [Review #{r_dict['id']}]: {r_dict['text']}\n"
        
    system_prompt = f"""You are an expert QA Engineer writing a consistent bug report based on user feedback.
Analyze the following reviews and write a comprehensive summary regarding: {topic_query}

IMPORTANT RULES:
1. You MUST format your response exactly using these three markdown headers:
   ### 📝 Overview
   ### 🚨 Key Issues
2. Use bullet points under Key Issues.
3. You MUST cite ALL 5 sources using the exact format [Review #ID] at the end of relevant sentences.
4. Be professional and objective.

Retrieved Reviews:
{reviews_text}"""

    ollama_urls = [
        "http://host.docker.internal:11434/api/generate",
        "http://localhost:11434/api/generate",
        "http://127.0.0.1:11434/api/generate"
    ]
    
    payload = {
        "model": "mistral",
        "prompt": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 500
        }
    }
    
    for url in ollama_urls:
        try:
            response = requests.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                answer = response.json().get("response", "").strip()
                return answer, top_reviews
        except requests.exceptions.RequestException:
            continue
            
    return "Error: Could not connect to Ollama. Please ensure it is running.", top_reviews

# ============================================================================
# EXTRACT CITATIONS
# ============================================================================
def extract_citations(text):
    """Extract unique [Review #XXXX] citations from text"""
    pattern = r'\[Review #(\d+)\]'
    matches = re.findall(pattern, text)
    # Convert to int and remove duplicates while preserving order
    return list(dict.fromkeys([int(m) for m in matches]))

# ============================================================================
# MAIN UI
# ============================================================================

tab1, tab2 = st.tabs(["💬 Interactive Q&A", "📊 Global Dataset Analysis"])

with tab1:
    st.subheader("🔍 Search & Analyze Reviews")
    st.write("Ask natural language questions about app reviews. Get answers backed by real user feedback with citations.")
    
    # Query input
    query = st.text_input(
        "❓ Enter your question:",
        placeholder="e.g., What do users complain about most? Why is battery drain an issue?",
        label_visibility="collapsed"
    )

    if query:
        st.markdown("---")
        
        # Retrieval phase
        with st.spinner("🔎 Searching for relevant reviews..."):
            top_indices = retrieve_reviews(query, top_k=5)
            
            top_reviews = []
            for idx in top_indices:
                top_reviews.append({
                    'id': review_ids[idx],
                    'rating': review_ratings[idx],
                    'date': review_dates[idx],
                    'text': review_texts[idx]
                })
        
        # Display retrieved reviews
        with st.expander("📋 Retrieved Reviews (5 most relevant)"):
            for i, review in enumerate(top_reviews, 1):
                col1, col2, col3 = st.columns([2, 1, 2])
                with col1:
                    st.markdown(f'<div id="review-{review["id"]}"></div>', unsafe_allow_html=True)
                    st.write(f"**Review #{review['id']}**")
                with col2:
                    rating_display = "⭐" * int(review['rating']) + "☆" * (5 - int(review['rating']))
                    st.write(rating_display)
                with col3:
                    st.caption(f"📅 {review['date']}")
                st.write(f"*{review['text'][:200]}...*" if len(review['text']) > 200 else f"*{review['text']}*")
                st.markdown("---")
        
        # Generation phase
        with st.spinner("✨ Generating answer with LLM..."):
            answer = generate_answer_with_citations(query, top_reviews)
        
        # Display answer
        st.markdown("---")
        st.subheader("💡 Answer")
        
        # Process answer to make citations clickable links to the chunks
        display_answer = answer
        citations = extract_citations(answer)
        for cid in citations:
            display_answer = display_answer.replace(f"[Review #{cid}]", f"[Review #{cid}](#review-{cid})")
            
        st.info(display_answer)
        
with tab2:
    st.subheader("📊 Global Dataset Analysis")
    st.write("Macro-level QA reports generated from the top 5 reviews for each category.")
    
    topics = {
        "🔋 Battery & Performance": "Summarize the main complaints about battery life, battery drain, device heating, and overall lag or performance issues.",
        "🎨 UI, UX & Navigation": "What are the common issues or feedback regarding the user interface, clunky navigation, unintuitive design, and visual bugs?",
        "💥 Crashes & Stability": "Summarize the major reports about app crashes, freezing, continuous loading screens, and general stability problems."
    }
    
    for title, topic_query in topics.items():
        with st.expander(title, expanded=True):
            with st.spinner(f"Generating macro report for {title}... (This will be cached)"):
                macro_summary, top_reviews_macro = generate_macro_topic_summary(topic_query)
                
                # Process citations for links
                display_summary = macro_summary
                citations = extract_citations(macro_summary)
                for cid in citations:
                    display_summary = display_summary.replace(f"[Review #{cid}]", f"[Review #{cid}](#macro-review-{cid})")
                    
                st.write(display_summary)
                
                # Render citations
                if citations:
                    st.markdown("##### 🔗 Source Evidence")
                    for citation_id in citations:
                        review = next((r for r in top_reviews_macro if r['id'] == citation_id), None)
                        if review:
                            with st.container(border=True):
                                st.markdown(f'<div id="macro-review-{citation_id}"></div>', unsafe_allow_html=True)
                                rating_display = "⭐" * int(review['rating']) + "☆" * (5 - int(review['rating']))
                                st.markdown(f"**Review #{citation_id}** | {rating_display} | 📅 {review['date']}")
                                st.write(f"*{review['text'][:150]}...*" if len(review['text']) > 150 else f"*{review['text']}*")
                        else:
                            st.warning(f"Review #{citation_id} not found in retrieved set.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
    <p>🤖 Powered by Ollama (mistral)</p>
    <p><small>Ensure Ollama is running correctly on your system.</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
