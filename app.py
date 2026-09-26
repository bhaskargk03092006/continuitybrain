import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json

# Set up page configurations
st.set_page_config(page_title="Continuity Brain - Sovereign AI", layout="wide")

# 1. Initialize Engines (Cached to save memory and processing time)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(name="continuity_brain")

model = load_embedding_model()
collection = get_chroma_collection()

# 2. Left Control Panel Sidebar Layout
st.sidebar.title("🧠 Continuity Brain v1.0")
st.sidebar.subheader("Track 1: Sovereign AI Engine")
st.sidebar.markdown("---")

# Visual Connection Status Indicators
st.sidebar.success("✅ ChromaDB Persistent State: Connected")
st.sidebar.success("✅ Local Embeddings Engine: Ready")

# Live metric calculation
try:
    total_logs = collection.count()
except Exception:
    total_logs = 0
st.sidebar.metric(label="Active Corporate Memory State", value=f"{total_logs} Decision Profiles")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Demo Strategy:** Turn off your WiFi completely to prove this app runs 100% locally with zero cloud dependencies.")

# 3. Main Interaction Field Panel Layout
st.title("Sovereign Executive Memory System")
st.markdown("##### Preserving corporate reasoning models and operational frameworks locally to empower coverage managers.")

# Risk Interceptor Configuration
HIGH_RISK_KEYWORDS = ["lawsuit", "terminate", "hack", "compromise", "illegal", "insolvent", "regulatory"]

user_input = st.text_area(
    "Describe the current corporate operational challenge or crisis configuration:",
    placeholder="e.g., A major competitor just undercut our core B2B pricing model by 30%. How should we address this market threat?",
    height=100
)

if st.button("Extract Executive Guidance Architecture"):
    if not user_input.strip():
        st.warning("Please enter a valid situational scenario to analyze.")
    else:
        # --- PHASE A: HIGH STAKES KEYWORD INTERCEPTOR ---
        matched_risks = [word for word in HIGH_RISK_KEYWORDS if word in user_input.lower()]
        if matched_risks:
            st.error(f"⚠️ **CRITICAL RISK WARNING:** High-stakes corporate liability triggers detected ({', '.join(matched_risks)}). All outputs must be cleared by legal compliance units before execution.")

        # --- PHASE B: SEMANTIC RETRIEVAL (Data -> Knowledge -> Memory) ---
        # Generate embedding for the new scenario query
        query_vector = model.encode(user_input).tolist()
        
        # Query top 3 nearest historical records
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )
        
        if results and results['ids'] and len(results['ids'][0]) > 0:
            # FIX: ChromaDB returns nested arrays [[val1, val2]]. Extract the exact 1st rank scalar float value.
            raw_distance = float(results['distances'][0][0])
            
            # Convert default Chroma L2 Squared Distance to a realistic 0-100 Similarity Percentage
            # (An L2 distance of 0.0 means perfect match, distances > 1.0 mean low similarity)
            similarity_score = max(0.0, min(100.0, (1.0 - (raw_distance / 2.0)) * 100.0))
            
            # --- PHASE C: THE DETERMINISTIC GUARDRAILS (Threshold Enforcer) ---
            if similarity_score < 50.0:
                st.error(
                    f"🚨 **Insufficient Historical Continuity Context ({similarity_score:.1f}% Match)**\n\n"
                    "No matching historical decisions exceed the mandatory 50% confidence baseline. "
                    "Do not rely on AI inference. Please scale this case manually to senior stakeholders or apply independent managerial discretion."
                )
            else:
                # FIX: Safely parse individual flat items from the matrix positions
                primary_doc = results['documents'][0][0]
                primary_meta = results['metadatas'][0][0]

            
                
                # --- PHASE D: LOCAL INFERENCE REASONING ENGINE (Reasoning -> Action) ---
                with st.spinner("Analyzing past historical reasoning via local Llama model..."):
                    # Construct strict prompt payload to control hallucination bounds
                    prompt_payload = f"""
                    You are acting as an executive context-continuity system. 
                    Based strictly on this past management decision profile, provide a synthesized response strategy for the new situation.
                    
                    PAST HISTORICAL EXPERIENCE:
                    Context Situation: {primary_doc}
                    Management Decision Taken: {primary_meta['decision']}
                    Underlying Rationale: {primary_meta['reasoning']}
                    Operational Outcome: {primary_meta['outcome']}
                    
                    NEW CURRENT SITUATION:
                    {user_input}
                    
                    Generate a clear guide keeping the same philosophical reasoning. 
                    Format your response with these exact headers:
                    ### 🎯 Suggested Strategic Stance
                    ### 💡 Inferred Executive Rationale
                    ### 📝 Draft Action Plan Response
                    """
                    
                    try:
                        # Direct HTTP Request hitting local Ollama API
                        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": "llama3.2",
                                "prompt": prompt_payload,
                                "stream": False
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            ai_synthesis = response.json().get("response", "Error parsing inference response bytes.")
                            st.subheader("🤖 Generated Executive Guidance Synthesis")
                            st.text_area("Review and refine this draft context:", value=ai_synthesis, height=350)
                        else:
                            st.error("⚠️ Local inference communication failed. Make sure your local Ollama server is active.")
                    except Exception as e:
                        st.error("⚠️ Unable to connect to local Ollama. Please check if 'ollama serve' is running in your terminal background.")
                
                # --- PHASE E: EXPERT AUDIT TRAIL DISPLAY ---
                st.markdown("---")
                with st.expander("🔍 Audit Trail Verification Metrics — Review Source Material Context"):
                    st.write(f"**Top Reference Identity Authenticated with a confidence rating of: `{similarity_score:.1f}%`**")
                    
                    # Create 3 layout blocks for top matches
                    cols = st.columns(3)
                    for i in range(len(results['ids'][0])):
                        if i < len(cols):
                            doc_text = results['documents'][0][i]
                            meta_data = results['metadatas'][0][i]
                            dist_val = results['distances'][0][i]
                            sim_pct = max(0.0, min(100.0, (1.0 - dist_val) * 100.0))
                            
                            with cols[i]:
                                st.markdown(f"##### Reference Match #{i+1} (`{sim_pct:.1f}%`)")
                                st.caption(f"**Past Situation:** {doc_text}")
                                st.markdown(f"**Decision:** {meta_data['decision']}")
                                st.markdown(f"**Reasoning:** {meta_data['reasoning']}")
                                st.markdown(f"**Outcome:** {meta_data['outcome']}")
        else:
            st.warning("The local memory collection is empty. Run `seed_db.py` to index management profiles.")
