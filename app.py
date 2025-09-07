"""
FAQ Chatbot Streamlit Application

A Streamlit-based FAQ chatbot using LangChain, FAISS, and Gemini API
with Retrieval-Augmented Generation (RAG) capabilities.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Optional
import time
import re
from collections import Counter

# Configure Streamlit page FIRST (before any other st commands)
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import advanced modules, fall back to simple approach if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from load_data import load_and_process_faqs
    from vectorstore import FAQVectorStore, create_faq_vectorstore, load_faq_vectorstore
    ADVANCED_MODE = True
except ImportError:
    ADVANCED_MODE = False

# Load environment variables
load_dotenv()

# Clean, Human-like CSS Styling
st.markdown("""
<style>
    /* Simple, clean fonts */
    @import url('https://fonts.googleapis.com/css2?family=System-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,sans-serif');
    
    /* Remove default Streamlit styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Simple header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #2d3748;
        text-align: center;
        margin-bottom: 1rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #718096;
        text-align: center;
        margin-bottom: 2rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Clean chat messages */
    .user-message {
        background: #3182ce;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px 60px;
        max-width: 70%;
        word-wrap: break-word;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .bot-message {
        background: #f7fafc;
        color: #2d3748;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 60px 8px 0;
        max-width: 70%;
        word-wrap: break-word;
        border: 1px solid #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .retrieved-docs {
        background: #fff5f5;
        border: 1px solid #fed7d7;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #744210;
    }
    
    /* Simple buttons */
    .stButton > button {
        background: #3182ce;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 14px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stButton > button:hover {
        background: #2c5aa0;
    }
    
    /* Clean input */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 1px solid #cbd5e0;
        padding: 12px 16px;
        font-size: 16px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3182ce;
        outline: none;
        box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
    }
    
    /* Simple sidebar */
    .css-1d391kg {
        background: #f7fafc;
    }
    
    /* Clean status cards */
    .status-card {
        background: white;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Simple sample buttons */
    .sample-btn {
        background: #f7fafc;
        color: #4a5568;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        border: 1px solid #e2e8f0;
        font-size: 14px;
        width: 100%;
        text-align: left;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .sample-btn:hover {
        background: #edf2f7;
        border-color: #cbd5e0;
    }
    
    /* Remove excessive spacing */
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
    
    /* Simple welcome message */
    .welcome-message {
        background: #f0fff4;
        border: 1px solid #9ae6b4;
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
        color: #22543d;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* --- Enhancements --- */
    .section-label {
        color: #2d3748;
        font-weight: 600;
        margin-top: 12px;
        margin-bottom: 6px;
    }
    .suggestions-wrap {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
    }
    .sample-btn {
        border-radius: 16px;
    }
    .status-card {
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .user-message {
        background: linear-gradient(135deg, #4c6fff, #3182ce);
        box-shadow: 0 8px 20px rgba(49,130,206,0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #4c6fff, #3182ce);
        box-shadow: 0 4px 12px rgba(49,130,206,0.22);
    }
    
    /* Sidebar polish */
    .sidebar-title {
        font-weight: 700;
        color: #1a202c;
        letter-spacing: -0.01em;
        margin: 6px 0 4px 0;
    }
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(203,213,224,0), rgba(203,213,224,0.9), rgba(203,213,224,0));
        margin: 10px 0;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 12px;
        border-radius: 9999px;
        border: 1px solid #e2e8f0;
        background: #f7fafc;
        color: #2d3748;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)


class FAQChatbot:
    """
    Hybrid FAQ Chatbot that works with both advanced and simple modes.
    """
    
    def __init__(self):
        self.vectorstore_manager = None
        self.llm = None
        self.llm_kind = None  # 'gemini' or 'palm'
        self.faq_data = []
        self.mode = "simple"  # "simple" or "advanced"
        self.setup()
    
    def setup(self):
        """Setup the chatbot based on available modules."""
        # Load FAQ data
        self.faq_data = self.load_faq_data()
        
        # Setup LLM
        self.setup_llm()
        
        # Determine mode
        if ADVANCED_MODE and self.faq_data:
            self.mode = "advanced"
            self.load_vectorstore()
        else:
            self.mode = "simple"
    
    def get_status(self) -> Dict[str, str]:
        """Expose minimal status for the sidebar."""
        return {
            'mode': self.mode,
            'llm': self.llm_kind or 'disabled',
            'faq_count': str(len(self.faq_data))
        }
    
    def reload_faqs(self) -> int:
        """Reload FAQs from file and return the new count."""
        self.faq_data = self.load_faq_data()
        return len(self.faq_data)
    
    def rebuild_index(self) -> bool:
        """Rebuild the vector index from current FAQs (advanced mode only)."""
        if not ADVANCED_MODE:
            return False
        try:
            documents = load_and_process_faqs("faqs.txt")
            self.vectorstore_manager = create_faq_vectorstore(documents)
            return True
        except Exception:
            return False
    
    def load_faq_data(self) -> List[Dict[str, str]]:
        """Load and parse FAQ data."""
        try:
            with open("faqs.txt", 'r', encoding='utf-8') as file:
                content = file.read()
            
            faqs = []
            sections = re.split(r'\n## ', content)
            
            for section in sections:
                if not section.strip():
                    continue
                    
                lines = section.strip().split('\n')
                category = lines[0].replace('#', '').strip()
                
                current_q = None
                current_a = []
                
                for line in lines[1:]:
                    line = line.strip()
                    if line.startswith('Q:'):
                        if current_q and current_a:
                            faqs.append({
                                'question': current_q,
                                'answer': ' '.join(current_a),
                                'category': category
                            })
                        current_q = line[2:].strip()
                        current_a = []
                    elif line.startswith('A:'):
                        current_a.append(line[2:].strip())
                    elif current_a and line:
                        current_a.append(line)
                
                if current_q and current_a:
                    faqs.append({
                        'question': current_q,
                        'answer': ' '.join(current_a),
                        'category': category
                    })
            
            return faqs
            
        except Exception as e:
            st.error(f"Error loading FAQ data: {str(e)}")
            return []
    
    def setup_llm(self):
        """Setup the language model."""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key and api_key != "your_gemini_api_key_here" and GEMINI_AVAILABLE:
                genai.configure(api_key=api_key)
                if hasattr(genai, 'GenerativeModel'):
                    # Newer google-generativeai
                    self.llm = genai.GenerativeModel(
                        'gemini-pro',
                        generation_config={
                            'temperature': 0.2,
                            'top_p': 0.9,
                            'top_k': 40,
                            'max_output_tokens': 512,
                        }
                    )
                    self.llm_kind = 'gemini'
                else:
                    # Older SDK fallback (PaLM Text Bison style)
                    self.llm = 'models/text-bison-001'
                    self.llm_kind = 'palm'
                # Silent success
            else:
                self.llm = None
                self.llm_kind = None
                # Silent fallback
        except Exception as e:
            # Silent failure
            self.llm = None
            self.llm_kind = None
    
    def load_vectorstore(self):
        """Load or create the FAQ vector store (advanced mode only)."""
        if not ADVANCED_MODE:
            return False
            
        try:
            # Try to load existing vector store
            if os.path.exists("faq_index"):
                self.vectorstore_manager = load_faq_vectorstore()
                st.success("✅ FAQ vector store loaded successfully")
            else:
                # Create new vector store
                st.info("🔄 Creating FAQ vector store...")
                documents = load_and_process_faqs("faqs.txt")
                self.vectorstore_manager = create_faq_vectorstore(documents)
                st.success("✅ FAQ vector store created successfully")
            return True
        except Exception as e:
            st.warning(f"⚠️ Vector store error: {str(e)}. Using simple mode.")
            return False
    
    def simple_search(self, query: str) -> List[Dict[str, str]]:
        """Improved keyword-based search with stopword filtering and weighting."""
        query_lower = query.lower()
        # Basic stopwords to avoid matching generic words
        stop = {
            'the','a','an','and','or','of','on','in','to','for','is','are','was','were','be','been','what','how','do','does','can','i','my','me','you','your','they','them','their','with','from','after','before','again'
        }
        query_terms = [w for w in re.findall(r"[a-z0-9']+", query_lower) if w not in stop]
        query_counts = Counter(query_terms)
        query_set = set(query_terms)
        # Lightweight keyword boosts for common intents
        boosts = set()
        for k in ["payment","payments","refund","return","exchange","delivery","track","order","coupon","offer","warranty","size","address"]:
            if k in query_set:
                boosts.add(k)
        results = []

        for faq in self.faq_data:
            question_terms = [w for w in re.findall(r"[a-z0-9']+", faq['question'].lower()) if w not in stop]
            answer_terms = [w for w in re.findall(r"[a-z0-9']+", faq['answer'].lower()) if w not in stop]

            question_set = set(question_terms)
            answer_set = set(answer_terms)

            # Heavier weight for overlaps with the FAQ question; lighter for answer
            question_overlap = sum(query_counts[t] for t in question_set if t in query_counts)
            answer_overlap = sum(query_counts[t] for t in answer_set if t in query_counts)

            score = (3.0 * question_overlap) + (0.25 * answer_overlap)

            # Small boost for exact phrase presence in question
            if query_lower.strip() and query_lower.strip() in faq['question'].lower():
                score += 2.0

            # Keyword boost if both query and question share a key intent term
            for b in boosts:
                if b in question_set:
                    score += 1.0

            # Penalize results with zero question overlap to reduce mismatches
            if question_overlap == 0:
                score *= 0.2

            if score > 0:
                results.append((score, faq))

        results.sort(key=lambda x: x[0], reverse=True)
        return [faq for score, faq in results[:5]]
    
    def generate_response_with_gemini(self, query: str, context_faqs: List[Dict[str, str]]) -> str:
        """Generate response using Gemini API."""
        try:
            # Prepare context
            context = "\n\n".join([
                f"Q: {faq['question']}\nA: {faq['answer']}" 
                for faq in context_faqs
            ])
            
            strict = st.session_state.get('strict_answers', True)
            policy = (
                "ONLY use the FAQ Context. If the answer is not present, say you don't have enough information."
                if strict else
                "Prefer the FAQ Context. If missing, you may answer with general guidance but clearly note uncertainty."
            )
            prompt = f"""
            You are a helpful FAQ assistant for Myntra.
            {policy}

            FAQ Context:
            {context}

            User Question: {query}

            Provide a short, accurate answer. Do not invent facts. If unsure, say so.
            """
            
            if self.llm_kind == 'gemini':
                response = self.llm.generate_content(prompt)
                return getattr(response, 'text', str(response))
            elif self.llm_kind == 'palm':
                response = genai.generate_text(model=self.llm, prompt=prompt)
                # Older SDK returns .result
                return getattr(response, 'result', str(response))
            else:
                raise RuntimeError('LLM not initialized')
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def generate_simple_response(self, query: str, context_faqs: List[Dict[str, str]]) -> str:
        """Generate a simple response without LLM."""
        if not context_faqs:
            return "I couldn't find any relevant information in our FAQ database. Please try rephrasing your question."
        
        # Find the most relevant FAQ
        best_match = context_faqs[0]
        
        return best_match['answer']
    
    def get_response(self, query: str) -> tuple:
        """Get response for a user query."""
        if self.mode == "advanced" and self.vectorstore_manager:
            try:
                # Use advanced vector search
                similar_docs = self.vectorstore_manager.search_similar_documents(query, k=8)
                if similar_docs:
                    context_faqs = []
                    for doc in similar_docs:
                        context_faqs.append({
                            'question': doc.metadata.get('question', ''),
                            'answer': doc.metadata.get('answer', ''),
                            'category': doc.metadata.get('category', '')
                        })
                else:
                    context_faqs = []
            except Exception as e:
                st.warning(f"Vector search error: {str(e)}. Using simple search...")
                context_faqs = self.simple_search(query)
        else:
            # Use simple search
            context_faqs = self.simple_search(query)
        
        if not context_faqs:
            return "I couldn't find any relevant information in our FAQ database. Please try rephrasing your question.", []
        
        # Generate response
        try:
            if self.llm:
                response = self.generate_response_with_gemini(query, context_faqs)
            else:
                response = self.generate_simple_response(query, context_faqs)
        except Exception as e:
            st.warning(f"LLM error: {str(e)}. Using simple response...")
            response = self.generate_simple_response(query, context_faqs)
        
        return response, context_faqs


def main():
    """Simple, clean chat interface."""
    
    # Simple header
    st.markdown('<h1 class="main-header">Myntra FAQ</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask me anything about Myntra</p>', unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading..."):
            st.session_state.chatbot = FAQChatbot()
    
    # Sidebar with status and quick actions
    with st.sidebar:
        status = st.session_state.chatbot.get_status()
        st.markdown('<div class="sidebar-title">Status</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div class='status-card'>"
            f"<div><strong>Mode</strong> <span class='badge'>{status['mode'].title()}</span></div>"
            f"<div style='margin-top:6px;'><strong>LLM</strong> <span class='badge'>{status['llm'].title()}</span></div>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("Clear chat"):
                st.session_state.chat_history = []
                st.rerun()
        with col_right:
            if st.button("Reload FAQs"):
                count = st.session_state.chatbot.reload_faqs()
                st.success(f"Reloaded FAQs ({count})")
                st.rerun()
        if st.button("Rebuild index"):
            ok = st.session_state.chatbot.rebuild_index()
            if ok:
                st.success("Rebuilt FAQ index")
            else:
                st.warning("Index rebuild unavailable in simple mode")
            st.rerun()
        st.toggle("Strict answers (only from FAQs)", key="strict_answers", value=True)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Quick topics</div>', unsafe_allow_html=True)
        for q in [
            "What is Myntra?",
            "Return policy basics",
            "Delivery timelines",
            "Accepted payment methods",
            "Track my order"
        ]:
            if st.button(q, key=f"quick_{q}"):
                st.session_state.chat_history.append({'role': 'user', 'content': q})
                with st.spinner("Thinking..."):
                    resp, src = st.session_state.chatbot.get_response(q)
                st.session_state.chat_history.append({'role': 'bot', 'content': resp, 'retrieved_faqs': src})
                st.rerun()
    
    # Chat area
    if not st.session_state.chatbot.faq_data:
        st.error("FAQ database not found")
        return
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Welcome message and Frequently Asked (one-click) only before first question
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="welcome-message">
            Hi! I can help with Myntra questions about returns, delivery, payments, and more.
        </div>
        """, unsafe_allow_html=True)

        frequently_asked = [
            "What is Myntra's return policy?",
            "How long does delivery take?",
            "What payment methods are accepted?",
            "How do I track my order?",
            "What if the product doesn't fit?"
        ]

        st.write("**Frequently asked**")
        col_a, col_b = st.columns(2)
        with col_a:
            for q in frequently_asked[::2]:
                if st.button(q, key=f"faq_{q}"):
                    # Process immediately
                    st.session_state.chat_history.append({'role': 'user', 'content': q})
                    with st.spinner("Thinking..."):
                        resp, src = st.session_state.chatbot.get_response(q)
                    st.session_state.chat_history.append({'role': 'bot', 'content': resp, 'retrieved_faqs': src})
                    st.rerun()
        with col_b:
            for q in frequently_asked[1::2]:
                if st.button(q, key=f"faq_{q}"):
                    st.session_state.chat_history.append({'role': 'user', 'content': q})
                    with st.spinner("Thinking..."):
                        resp, src = st.session_state.chatbot.get_response(q)
                    st.session_state.chat_history.append({'role': 'bot', 'content': resp, 'retrieved_faqs': src})
                    st.rerun()
    
    # Show messages
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
            
            # (Sources hidden as requested)
    
    # Dynamic follow-up suggestions after last bot reply
    def build_suggestions() -> List[str]:
        if not st.session_state.chat_history:
            return []
        last_bot = None
        for item in reversed(st.session_state.chat_history):
            if item.get('role') == 'bot':
                last_bot = item
                break
        suggestions: List[str] = []
        if last_bot and last_bot.get('retrieved_faqs'):
            seen = set()
            for f in last_bot['retrieved_faqs']:
                q = f.get('question', '').strip()
                if q and q not in seen:
                    seen.add(q)
                    suggestions.append(q)
                if len(suggestions) >= 4:
                    break
        if not suggestions:
            suggestions = [
                "What is Myntra?",
                "How do returns work?",
                "What are delivery timelines?",
                "Which payment methods are supported?",
            ]
        last_user = None
        for item in reversed(st.session_state.chat_history):
            if item.get('role') == 'user':
                last_user = item.get('content')
                break
        if last_user:
            suggestions = [s for s in suggestions if s.strip().lower() != last_user.strip().lower()]
        return suggestions[:4]

    suggestions = build_suggestions()
    if suggestions:
        st.markdown('<div class="section-label">Try one of these</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for idx, s in enumerate(suggestions):
            with cols[idx % 2]:
                if st.button(s, key=f"sugg_{idx}"):
                    st.session_state.chat_history.append({'role': 'user', 'content': s})
                    with st.spinner("Thinking..."):
                        resp, src = st.session_state.chatbot.get_response(s)
                    st.session_state.chat_history.append({'role': 'bot', 'content': resp, 'retrieved_faqs': src})
                    st.rerun()
    
    # Input area
    st.write("---")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question:",
            value=st.session_state.get('user_question', ''),
            placeholder="Type your question here...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_btn = st.button("Send", type="primary")
    
    # Process question
    if send_btn and user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Get response
        with st.spinner("Thinking..."):
            response, sources = st.session_state.chatbot.get_response(user_input)
        
        # Add bot response
        st.session_state.chat_history.append({
            'role': 'bot',
            'content': response,
            'retrieved_faqs': sources
        })
        
        # Clear input
        st.session_state.user_question = ""
        st.rerun()


if __name__ == "__main__":
    main()