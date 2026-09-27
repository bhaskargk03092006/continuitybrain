import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json

st.set_page_config(page_title="Continuity Brain - Sovereign AI", layout="wide")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(name="continuity_brain")

model = load_embedding_model()
collection = get_chroma_collection()

# --- SIDEBAR DESIGN ---
st.sidebar.title("🧠 Continuity Brain v1.0")
st.sidebar.subheader("Track 1: Sovereign AI Engine")
st.sidebar.markdown("---")
st.sidebar.success("✅ ChromaDB Storage: Local Local")
st.sidebar.success("✅ Embeddings Transformer: Offline")

try:
    total_logs = collection.count()
except Exception:
    total_logs = 0
st.sidebar.metric(label="Total Executive Memories Saved", value=f"{total_logs} Profiles")
st.sidebar.markdown("---")
st.sidebar.warning("⚡ **Pitch Strategy:** Turn off Wi-Fi in front of the judges to prove 100% data sovereignty!")

# --- MAIN APP LAYOUT USING TABS ---
st.title("Sovereign Executive Memory System")
tab1, tab2 = st.tabs(["🔍 Query Executive Brain", "📝 Log New Executive Decision"])

# --- TAB 1: SEARCH & REASONING PIPELINE ---
with tab1:
    st.markdown("##### Query historical institutional memory to align emergency decisions with corporate philosophy.")
    HIGH_RISK_KEYWORDS = ["lawsuit", "terminate", "hack", "compromise", "illegal", "insolvent", "regulatory"]
    
    user_input = st.text_area(
        "Describe the current corporate operational challenge or crisis configuration:",
        placeholder="e.g., A major competitor just undercut our pricing model by 30%. How should we handle this?",
        key="search_input"
    )
    
    if st.button("Extract Executive Guidance Architecture"):
        if not user_input.strip():
            st.warning("Please enter a valid scenario.")
        else:
            matched_risks = [word for word in HIGH_RISK_KEYWORDS if word in user_input.lower()]
            if matched_risks:
                st.error(f"⚠️ **CRITICAL RISK WARNING:** High-stakes elements detected ({', '.join(matched_risks)}). Output must be vetted by Legal.")

            query_vector = model.encode(user_input).tolist()
            results = collection.query(query_embeddings=[query_vector], n_results=3)
            
            if results and results['ids'] and len(results['ids'][0]) > 0:
                raw_distance = float(results['distances'][0][0])
                similarity_score = max(0.0, min(100.0, (1.0 - (raw_distance / 2.0)) * 100.0))
                
                if similarity_score < 50.0:
                    st.error(f"🚨 **Insufficient Historical Continuity Context ({similarity_score:.1f}% Match)**\n\nFallback to manual stakeholder escalation enforced.")
                else:
                    primary_doc = results['documents'][0][0]
                    primary_meta = results['metadatas'][0][0]
                    
                    with st.spinner("Synthesizing guidance via local Llama model..."):
                        prompt_payload = f"""
                        You are an executive context-continuity system. 
                        Based strictly on this past management decision, provide a strategy for the new situation.
                        
                        PAST EXPERIENCE:
                        Situation: {primary_doc}
                        Decision: {primary_meta['decision']}
                        Rationale: {primary_meta['reasoning']}
                        Outcome: {primary_meta['outcome']}
                        
                        NEW SITUATION:
                        {user_input}
                        
                        Format response with:
                        ### 🎯 Suggested Strategic Stance
                        ### 💡 Inferred Executive Rationale
                        ### 📝 Draft Action Plan Response
                        """
                        try:
                            response = requests.post(
                                "http://localhost:11434/api/generate",
                                json={"model": "llama3.2", "prompt": prompt_payload, "stream": False},
                                timeout=30
                            )
                            if response.status_code == 200:
                                ai_synthesis = response.json().get("response", "Error parsing response.")
                                st.subheader("🤖 Generated Executive Guidance Synthesis")
                                st.text_area("Review and copy draft output:", value=ai_synthesis, height=300)
                            else:
                                st.error("⚠️ Local inference failure. Check Ollama server status.")
                        except Exception:
                            st.error("⚠️ Connection to local Ollama failed. Run 'ollama serve' in background.")
                    
                    # AUDIT TRAIL
                    st.markdown("---")
                    with st.expander("🔍 Audit Trail Verification Metrics"):
                        cols = st.columns(3)
                        for i in range(len(results['ids'][0])):
                            doc_text = results['documents'][0][i]
                            meta_data = results['metadatas'][0][i]
                            dist_val = results['distances'][0][i]
                            sim_pct = max(0.0, min(100.0, (1.0 - (dist_val / 2.0)) * 100.0))
                            with cols[i]:
                                st.markdown(f"##### Match #{i+1} (`{sim_pct:.1f}%`)")
                                st.caption(f"**Past Context:** {doc_text}")
                                st.markdown(f"**Decision:** {meta_data['decision']}")
                                st.markdown(f"**Reasoning:** {meta_data['reasoning']}")

# --- TAB 2: LIVE DATA INGESTION ENGINE ---
with tab2:
    st.markdown("##### Expand the Sovereign Brain by logging real-time executive framework profiles.")
    
    with st.form("new_decision_form"):
        new_sit = st.text_area("1. The Corporate Situation / Context:", placeholder="Describe the crisis or scenario faced...")
        new_dec = st.text_area("2. The Management Decision Taken:", placeholder="What explicit operational step was executed?")
        new_reas = st.text_area("3. Core Underlying Rationale:", placeholder="Why was this specific decision preferred over options?")
        new_out = st.text_area("4. Measured Operational Outcome:", placeholder="What was the short/long term business impact?")
        
        submit_btn = st.form_submit_button("Commit Frame to Persistent Memory")
        
        if submit_btn:
            if not (new_sit.strip() and new_dec.strip() and new_reas.strip() and new_out.strip()):
                st.error("❌ All four operational metrics must be filled out before saving.")
            else:
                # Vectorize raw run-time user input text data block
                new_vector = model.encode(new_sit).tolist()
                new_id = f"doc_id_runtime_{collection.count() + 1}"
                
                collection.add(
                    documents=[new_sit],
                    embeddings=[new_vector],
                    metadatas=[{"decision": new_dec, "reasoning": new_reas, "outcome": new_out}],
                    ids=[new_id]
                )
                st.success(f"✅ Record successfully authenticated and indexed into ChromaDB storage as reference node: `{new_id}`!")
                st.rerun()
