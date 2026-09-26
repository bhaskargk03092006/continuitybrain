import chromadb
from sentence_transformers import SentenceTransformer
import os

def initialize_database():
    print("🤖 Step 1: Initializing Local Embedding Model (all-MiniLM-L6-v2)...")
    # This downloads once when online, and runs locally afterwards
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("📦 Step 2: Setting up Local Persistent ChromaDB Client...")
    # Creates a physical folder './chroma_db' in your project directory
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or fetch existing collection
    collection = chroma_client.get_or_create_collection(name="continuity_brain")
    
    print("📝 Step 3: Preparing Synthetic Executive Decision Logs...")
    # Master structure: Situation | Decision | Reasoning | Outcome
    decision_logs = [
        {
            "situation": "Major production database failure during peak traffic hours causing 40% user drop.",
            "decision": "Rolled back the last microservice deployment immediately and initiated an offline database replication strategy.",
            "reasoning": "Prioritized active user session retention and platform stabilization over immediate bug resolution. Root cause analysis could be done post-stabilization.",
            "outcome": "Platform stabilized within 14 minutes; remaining data synced successfully overnight without customer friction."
        },
        {
            "situation": "Critical customer support head resigned unexpectedly right before a major enterprise client onboarding sprint.",
            "decision": "Temporarily reallocated two senior product managers to handle direct client communications and account onboarding.",
            "reasoning": "Enterprise clients value domain and product expertise during onboarding. Product managers possess the deepest knowledge to cover tactical support gaps.",
            "outcome": "Onboarding achieved a 100% satisfaction score; cross-departmental exposure improved the PM feature pipeline."
        },
        {
            "situation": "Competitor launched an aggressive pricing strategy, undercutting our core B2B SaaS subscription tier by 30%.",
            "decision": "Maintained existing premium pricing tier but bundled dedicated security compliance and priority support access for free.",
            "reasoning": "Engaging in a race-to-the-bottom price war erodes brand value. Enterprise buyers value stability, compliance, and guaranteed support over pure cost reductions.",
            "outcome": "Customer churn held below 2% and average contract value actually increased due to premium feature visibility."
        },
        {
            "situation": "Internal engineering team requests a complete 3-week sprint pause to address massive accumulated technical debt.",
            "decision": "Denied the absolute pause. Allocated a fixed 20% allocation capacity of every subsequent sprint exclusively to tech debt resolution.",
            "reasoning": "A complete feature freeze kills market momentum and damages the investor milestone pipeline. Continuous micro-refactoring balances velocity with systemic health.",
            "outcome": "Critical technical blockages resolved over 2 months without delaying major commercial release commitments."
        }
    ]
    
    print(f"🚀 Step 4: Encoding and upserting {len(decision_logs)} entries to local ChromaDB...")
    
    for idx, log in enumerate(decision_logs):
        # We embed the "situation" text to match user crisis situations later
        embedding = model.encode(log["situation"]).tolist()
        
        # Metadata keeps the rest of the operational parameters accessible
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
