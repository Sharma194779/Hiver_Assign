import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from typing import List, Tuple, Dict
import json
import re

class MiniRAG:
    def __init__(self, 
                 embedding_model="BAAI/bge-small-en-v1.5",
                 llm_model="microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize Mini-RAG system with embeddings and LLM
        
        Args:
            embedding_model: Model for creating embeddings
            llm_model: LLM for answer generation
        """
        print("Loading RAG components...")
        
        print(f"  Loading embeddings: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        print(f"  Loading LLM: {llm_model}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.index = None
        self.documents = []
        self.doc_metadata = []
        
        print("RAG system initialized!")
    
    def chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        """Split text into smaller chunks for better embeddings"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks if chunks else [text]
    
    def load_kb_articles(self, kb_folder: str):
        """Load knowledge base articles from folder with chunking"""
        print(f"\nLoading KB articles from {kb_folder}...")
        
        article_count = 0
        chunk_count = 0
        
        for filename in os.listdir(kb_folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(kb_folder, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    chunks = self.chunk_text(content, chunk_size=300)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        self.documents.append(chunk)
                        self.doc_metadata.append({
                            'filename': filename,
                            'chunk_id': chunk_idx,
                            'length': len(chunk)
                        })
                        chunk_count += 1
                    
                    article_count += 1
        
        print(f"  Loaded {article_count} articles ({chunk_count} chunks)")
    
    def build_index(self):
        """Build FAISS vector index from documents"""
        print("\nBuilding vector index...")
        
        embeddings = self.embedder.encode(
            self.documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors (dim={dimension})")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve most relevant documents for query
        
        Returns:
            List of (document, distance, metadata) tuples
        """
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((
                self.documents[idx],
                float(dist),
                self.doc_metadata[idx]
            ))
        
        return results
    
    def rerank(self, query: str, retrieved_docs: List[str]) -> Tuple[List[str], List[float]]:
        """
        Rerank retrieved documents using cosine similarity
        
        Returns:
            Tuple of (reranked_docs, similarity_scores)
        """
        q_emb = self.embedder.encode(query, convert_to_numpy=True)
        
        scores = []
        for doc in retrieved_docs:
            d_emb = self.embedder.encode(doc, convert_to_numpy=True)
            similarity = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-8)
            scores.append(float(similarity))
        
        sorted_pairs = sorted(zip(scores, retrieved_docs), reverse=True)
        reranked_scores = [score for score, _ in sorted_pairs]
        reranked_docs = [doc for _, doc in sorted_pairs]
        
        return reranked_docs, reranked_scores
    
    def calculate_confidence(self, similarities: List[float]) -> float:
        """
        Calculate confidence score based on cosine similarities
        
        Args:
            similarities: List of cosine similarity scores (-1 to 1)
        
        Returns:
            Confidence score (0 to 1)
        """
        if not similarities:
            return 0.0
        
        sim_norm = [(s + 1) / 2 for s in similarities]
        return float(np.mean(sim_norm))
    
    def generate_answer(self, query: str, context_docs: List[str]) -> str:
        """Generate answer using LLM with retrieved context"""
        
        context = "\n\n".join([f"Context {i+1}:\n{doc[:400]}" for i, doc in enumerate(context_docs[:3])])
        
        prompt = f"""You are a strict RAG assistant for Hiver Support.

Rules:
1. You MUST answer ONLY using the provided Knowledge Base context.
2. If the context does not contain enough information, answer:
   "The knowledge base does not contain information about this."
3. Do NOT add additional facts.
4. Provide a short, accurate answer.

Knowledge Base Context:
{context}

User Question:
{query}

Answer (based ONLY on the context):"""
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1800
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        match = re.search(r"Answer\s*\(based ONLY on the context\):\s*(.*)", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
        else:
            answer = response.split("Answer")[-1].strip()
            answer = re.sub(r'^\(based ONLY on the context\):\s*', '', answer)
        
        answer = answer.split('\n')[0] if '\n' in answer else answer
        
        return answer
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Full RAG pipeline: retrieve, rerank, and generate
        
        Returns:
            Dictionary with retrieved articles, answer, confidence, and sources
        """
        initial_results = self.retrieve(question, top_k=top_k)
        
        docs = [doc for doc, _, _ in initial_results]
        distances = [dist for _, dist, _ in initial_results]
        metadata = [meta for _, _, meta in initial_results]
        
        reranked_docs, similarities = self.rerank(question, docs)
        
        confidence = self.calculate_confidence(similarities)
        
        answer = self.generate_answer(question, reranked_docs[:3])
        
        unique_sources = list(dict.fromkeys([meta['filename'] for meta in metadata]))
        
        return {
            'question': question,
            'top_k': top_k,
            'retrieved_docs': [
                {
                    'content': doc[:250] + "...",
                    'distance': dist,
                    'similarity': sim,
                    'metadata': meta
                }
                for doc, dist, sim, meta in zip(docs, distances, similarities, metadata)
            ],
            'reranked_docs': [
                {
                    'content': doc[:250] + "...",
                    'similarity': sim
                }
                for doc, sim in zip(reranked_docs[:3], similarities[:3])
            ],
            'answer': answer,
            'confidence': round(confidence, 3),
            'sources': unique_sources
        }

def create_sample_kb_articles():
    """Create sample KB articles for testing"""
    os.makedirs('data/kb_articles', exist_ok=True)
    
    articles = {
        'automation_setup.txt': """How to Configure Automations in Hiver

Automations in Hiver help you streamline repetitive tasks. Here's how to set them up:

1. Navigate to Settings > Automations
2. Click "Create New Automation"
3. Define triggers (e.g., email received, tag added)
4. Set conditions (e.g., subject contains, from specific sender)
5. Choose actions (e.g., assign to team member, add tag, send notification)
6. Test your automation with sample emails
7. Activate the automation

Best Practices:
- Start with simple rules and iterate
- Use specific conditions to avoid false triggers
- Monitor automation logs regularly
- Disable automations when debugging issues

Common Issues:
- Automations not firing: Check conditions are met
- Duplicate actions: Review multiple automation rules
- Delays in execution: Normal processing can take 1-2 minutes""",

        'csat_analytics.txt': """CSAT (Customer Satisfaction) Analytics in Hiver

Understanding your CSAT metrics:

What is CSAT?
Customer Satisfaction Score measures customer happiness after interactions.

Where to find CSAT:
- Dashboard > Analytics > CSAT
- Individual conversation views show CSAT ratings
- Export reports via Analytics > Export

CSAT Not Appearing? Troubleshooting:
1. Verify CSAT surveys are enabled in Settings
2. Check if surveys are being sent (Automation logs)
3. Ensure email templates include CSAT links
4. Verify customers are clicking survey links
5. Check date range filters in Analytics dashboard

Common Causes:
- CSAT disabled after trial period
- Survey sending automation paused
- Email client blocking survey links
- Analytics sync delay (updates hourly)

If CSAT still not visible:
- Clear browser cache
- Wait for next hourly sync
- Contact support for account-specific issues""",

        'email_assignment.txt': """Email Assignment in Hiver

How email assignment works:

Manual Assignment:
- Open email conversation
- Click "Assign" button
- Select team member
- Add optional note

Auto-Assignment:
- Set up automation rules
- Define assignment criteria (tags, keywords, senders)
- Choose round-robin or load-based distribution

Assignment Issues:
- Emails reverting to unassigned: Check for conflicting automations
- Wrong agent assigned: Review automation conditions
- Assignment delays: Normal processing can take 1-2 minutes

Best Practices:
- Use tags to categorize before auto-assignment
- Set up backup assignment rules
- Monitor unassigned email queue regularly""",

        'sla_configuration.txt': """Setting Up SLAs in Hiver

Service Level Agreements (SLAs) help track response times:

SLA Setup Steps:
1. Go to Settings > SLAs
2. Create SLA policy
3. Define response time targets (e.g., 2 hours first response)
4. Set resolution time goals
5. Choose which emails to apply SLA to (tags, customers, priority)
6. Configure escalation rules
7. Enable notifications for approaching/breached SLAs

Different Customer Tiers:
- Create separate SLA policies for VIP vs standard customers
- Use tags to differentiate customer tiers
- Set priority levels (high, medium, low)
- Configure different time targets per tier

SLA Monitoring:
- Dashboard shows SLA compliance percentage
- Individual emails show SLA status
- Reports available for historical analysis""",

        'workflow_troubleshooting.txt': """Workflow and Rules Troubleshooting

Common workflow issues and solutions:

Rules Not Triggering:
1. Verify rule is enabled (Settings > Workflows)
2. Check conditions match email attributes exactly
3. Review order of rules (first match wins)
4. Check for typos in subject/body matching
5. Ensure rule hasn't been paused

Workflow Delays:
- Normal processing: 1-3 minutes
- High volume: May take up to 5 minutes
- Check system status for outages

Rules Not Saving:
- Ensure all required fields filled
- Check for special characters causing issues
- Try different browser
- Clear cache and retry

Testing Workflows:
- Use "Test" mode with sample emails
- Monitor automation logs
- Start with one simple rule
- Add complexity gradually

Workflow Best Practices:
- Document each rule's purpose
- Review rules monthly
- Remove unused rules
- Use specific conditions over broad matches"""
    }
    
    for filename, content in articles.items():
        with open(f'data/kb_articles/{filename}', 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created {len(articles)} sample KB articles")

def main():
    """Main execution for Part C"""
    
    create_sample_kb_articles()
    
    rag = MiniRAG()
    
    rag.load_kb_articles('data/kb_articles')
    
    rag.build_index()
    
    queries = [
        "How do I configure automations in Hiver?",
        "Why is CSAT not appearing in my dashboard?"
    ]
    
    results = []
    
    print("\n" + "="*60)
    for query_text in queries:
        print(f"\nQuery: {query_text}")
        print("-" * 60)
        
        result = rag.query(query_text, top_k=5)
        
        print(f"\nRetrieved and Reranked Articles:")
        for i, doc_info in enumerate(result['reranked_docs'], 1):
            print(f"  {i}. Similarity: {doc_info['similarity']:.3f}")
            print(f"     Content: {doc_info['content']}")
        
        print(f"\nGenerated Answer:")
        print(f"  {result['answer']}")
        
        print(f"\nConfidence Score: {result['confidence']:.2%}")
        print(f"Sources: {', '.join(result['sources'])}")
        
        results.append(result)
        print("="*60)
    
    with open('outputs/part_c_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nPart C complete! Results saved to outputs/part_c_results.json")

if __name__ == "__main__":
    main()
