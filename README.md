# 🧠 Continuity Brain — Sovereign AI Engine

An offline-first, local corporate memory and decision-support architecture built for the **ASYNC'26 Hackathon (Track 1 Sovereign AI)**. 

## 🚀 The Core Idea
When key decision-makers (founders/managers) are unavailable, organizations face context gaps. **Continuity Brain** maps out institutional memory using an offline RAG framework. It matches current crisis scenarios with past executive decisions using **Situation | Decision | Reasoning | Outcome** metrics, allowing junior managers to query how their leads would approach a situation.

## 🛠️ High-Performance Architecture (Data ➡️ Knowledge ➡️ Memory ➡️ Reasoning ➡️ Action)
1. **Data:** Raw transactional data framed inside granular corporate decision schemas.
2. **Knowledge:** Relational indices embedded locally.
3. **Memory:** Stored securely within a local persistent vector space (**ChromaDB**).
4. **Reasoning:** Local context expansion using **Ollama** + **Llama 3.2 3B**.
5. **Action:** Secure generation of strategic response drafts along with full deterministic compliance verification.

## 🔒 Deterministic Sovereign Guardrails
* **50% Confidence Threshold:** If the semantic match rating falls below 50%, the LLM is blocked, forcing safe human escalation.
* **Risk Interceptor Layer:** Instantly catches high-stakes keywords (e.g., *lawsuit, hack, terminate*) and flags liability warnings.
* **100% Data Sovereignty:** Zero cloud data transfers. Runs natively on device, validated through offline execution.

## 📦 Local Installation & Setup

1. **Clone the repository structure:**
   ```bash
   git clone https://github.com
   cd continuitybrain
   ```

2. **Install local dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize and Seed the Local Memory DB:**
   ```bash
   python seed_db.py
   ```

4. **Boot up local inference hosting:**
   Ensure Ollama is running on your machine, then execute:
   ```bash
   ollama pull llama3.2
   ```

5. **Launch the presentation portal:**
   ```bash
   streamlit run app.py
   ```
