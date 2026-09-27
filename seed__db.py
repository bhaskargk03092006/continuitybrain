import chromadb
from sentence_transformers import SentenceTransformer
import os

def initialize_database():
    print("🤖 Step 1: Initializing Local Embedding Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("📦 Step 2: Setting up Local Persistent ChromaDB Client...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Purane database ko reset karne ke liye collection delete karke naya banate hain
    try:
        chroma_client.delete_collection(name="continuity_brain")
        print("🗑️ Cleared older database logs for fresh build.")
    except Exception:
        pass
        
    collection = chroma_client.get_or_create_collection(name="continuity_brain")
    
    print("📝 Step 3: Preparing Complete Synthetic Executive Decision Logs...")
    decision_logs = [
        # --- TECH & PRODUCT STRATEGY ---
        {
            "situation": "Major production database failure during peak traffic hours causing 40% user drop.",
            "decision": "Rolled back the last microservice deployment immediately and initiated an offline database replication strategy.",
            "reasoning": "Prioritized active user session retention and platform stabilization over immediate bug resolution. Root cause analysis could be done post-stabilization.",
            "outcome": "Platform stabilized within 14 minutes; remaining data synced successfully overnight without customer friction."
        },
        {
            "situation": "Internal engineering team requests a complete 3-week sprint pause to address massive accumulated technical debt.",
            "decision": "Denied the absolute pause. Allocated a fixed 20% allocation capacity of every subsequent sprint exclusively to tech debt resolution.",
            "reasoning": "A complete feature freeze kills market momentum and damages the investor milestone pipeline. Continuous micro-refactoring balances velocity with systemic health.",
            "outcome": "Critical technical blockages resolved over 2 months without delaying major commercial release commitments."
        },
        {
            "situation": "Core open-source library used in our authentication module announced it is deprecating free tier usage terms.",
            "decision": "Migrated entire authentication framework to a fully self-hosted, fork-maintained community version overnight.",
            "reasoning": "Sovereign infrastructure means zero vendor lock-in for critical utility layers. Paying sudden premium licenses spikes operating costs unpredictably.",
            "outcome": "Authentication migration successful with zero user downtime and exactly $0 added external platform billing liability."
        },
        # --- CRISIS & PR MANAGEMENT ---
        {
            "situation": "A junior developer accidentally exposed a non-production staging API key on a public GitHub repository for 6 hours.",
            "decision": "Revoked the API token instantly, forced a global environment variable rotation, and initiated an audit log review.",
            "reasoning": "Immediate containment minimizes exploit windows. Pre-emptive token rotation neutralizes structural vulnerabilities before they can be leveraged by external scans.",
            "outcome": "Zero unauthorized data access signatures detected in log audits. System secured within 45 minutes of discovery."
        },
        {
            "situation": "A vocal ex-employee posted highly critical, misleading reviews on Glassdoor claiming toxic management behavior.",
            "decision": "Issued a neutral corporate response acknowledging feedback while choosing not to argue specific claims publicly.",
            "reasoning": "Engaging in public emotional arguments amplifies negative PR. A professional, measured stance positions the leadership team as mature and level-headed.",
            "outcome": "The thread died down within 48 hours; overall company brand rating remained steady without investor panic."
        },
        # --- HR & TEAM DYNAMICS ---
        {
            "situation": "Critical customer support head resigned unexpectedly right before a major enterprise client onboarding sprint.",
            "decision": "Temporarily reallocated two senior product managers to handle direct client communications and account onboarding.",
            "reasoning": "Enterprise clients value domain and product expertise during onboarding. Product managers possess the deepest knowledge to cover tactical support gaps.",
            "outcome": "Onboarding achieved a 100% satisfaction score; cross-departmental exposure improved the PM feature pipeline."
        },
        {
            "situation": "Two core senior engineers had a severe professional disagreement over architectural design, threatening to quit.",
            "decision": "Appointed a neutral external technical consultant to audit both designs and choose the optimal path objectively.",
            "reasoning": "Removing internal ego from technical bottlenecks preserves team cohesion. Data-driven external arbitration provides an objective exit strategy for conflicts.",
            "outcome": "Consultant selected a hybrid design; both engineers felt respected and remained with the organization."
        },
        # --- FINANCE & BURN RATE ---
        {
            "situation": "Competitor launched an aggressive pricing strategy, undercutting our core B2B SaaS subscription tier by 30%.",
            "decision": "Maintained existing premium pricing tier but bundled dedicated security compliance and priority support access for free.",
            "reasoning": "Engaging in a race-to-the-bottom price war erodes brand value. Enterprise buyers value stability, compliance, and guaranteed support over pure cost reductions.",
            "outcome": "Customer churn held below 2% and average contract value actually increased due to premium feature visibility."
        },
        {
            "situation": "Main venture capital investor delayed our bridge funding round by 60 days due to internal compliance bottlenecks.",
            "decision": "Cut non-essential software SaaS subscriptions and deferred founders' personal salaries to preserve 3 months of runway.",
            "reasoning": "Cash runway preservation is paramount during macro funding pauses. Safeguarding active employee payroll takes priority over founder compensation.",
            "outcome": "Company maintained operational capability for 60 days until the bridge funding wire cleared cleanly."
        }
    ]
    
    print(f"🚀 Step 4: Encoding and upserting {len(decision_logs)} entries to local ChromaDB...")
    for idx, log in enumerate(decision_logs):
        embedding = model.encode(log["situation"]).tolist()
        metadata = {
            "decision": log["decision"],
            "reasoning": log["reasoning"],
            "outcome": log["outcome"]
        }
        collection.add(
            documents=[log["situation"]],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[f"doc_id_{idx}"]
        )
    print("✅ Database successfully seeded and compiled locally!")

if __name__ == "__main__":
    initialize_database()
