from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Tuple
import tensorflow as tf
from dataclasses import dataclass
from html_parser import BackendNodeParser, Node  # Assuming this is your existing parser

@dataclass
class RankedNode:
    node: Node
    score: float
    match_type: str  # 'text', 'context', or 'combined'

class HTMLRAG:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.parser = BackendNodeParser()
        self.node_embeddings: Dict[str, Dict] = {}
        
    def process_html(self, html_content: str):
        """Process HTML content and create embeddings for all nodes"""
        self.parser.parse_html(html_content)
        self._create_embeddings()

    def _create_embeddings(self):
        """Create embeddings for all interactive nodes"""
        for node_id, node in self.parser.nodes.items():
            if node.is_interactive:
                # Create separate embeddings for text content and context
                text_embedding = self._embed_text(node.text_content)
                context = self.parser.get_node_context(node_id)
                context_embedding = self._embed_text(context)
                
                # Store embeddings and original text
                self.node_embeddings[node_id] = {
                    'text_embedding': text_embedding,
                    'context_embedding': context_embedding,
                    'combined_embedding': (text_embedding + context_embedding) / 2,
                    'text': node.text_content,
                    'context': context
                }

    def _embed_text(self, text: str) -> np.ndarray:
        """Create embedding for a given text"""
        if not text.strip():
            # Return zero vector if text is empty
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode(text, convert_to_tensor=True).cpu().numpy()

    def find_relevant_nodes(self, 
                           query: str, 
                           top_k: int = 5,
                           threshold: float = 0.3) -> List[RankedNode]:
        """
        Find relevant nodes based on a query
        
        Args:
            query: The search query
            top_k: Number of results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of RankedNode objects
        """
        query_embedding = self._embed_text(query)
        scores = []
        
        for node_id, embeddings in self.node_embeddings.items():
            node = self.parser.nodes[node_id]
            
            # Calculate similarity scores
            text_score = self._calculate_similarity(query_embedding, embeddings['text_embedding'])
            context_score = self._calculate_similarity(query_embedding, embeddings['context_embedding'])
            combined_score = self._calculate_similarity(query_embedding, embeddings['combined_embedding'])
            
            # Determine best match type
            if text_score >= context_score and text_score >= combined_score:
                best_score = text_score
                match_type = 'text'
            elif context_score >= text_score and context_score >= combined_score:
                best_score = context_score
                match_type = 'context'
            else:
                best_score = combined_score
                match_type = 'combined'
                        
            if best_score >= threshold:
                scores.append(RankedNode(node=node, score=best_score, match_type=match_type))
        
        # Sort by score and return top k
        return sorted(scores, key=lambda x: x.score, reverse=True)[:top_k]

    def _calculate_similarity(self, query_embedding: np.ndarray, node_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between query and node embeddings"""
        top_values, top_indices = tf.math.top_k(node_embedding, k=5)

    def get_context_for_node(self, node_id: str) -> str:
        """Get the stored context for a node"""
        if node_id in self.node_embeddings:
            return self.node_embeddings[node_id]['context']
        return ""

def test_html_rag():
    html_content = """
    <div backend_node_id="1">
        <button backend_node_id="2" class="primary">Sign up for newsletter</button>
        <a backend_node_id="3" href="#" aria-label="Navigate to home page">Home</a>
        <div backend_node_id="4">
            <input backend_node_id="5" type="text" placeholder="Search for products">
            <button backend_node_id="6">Search</button>
        </div>
        <div backend_node_id="7">
            <button backend_node_id="8">Add to cart</button>
            <p backend_node_id="9">Free shipping on orders over $50</p>
        </div>
    </div>
    """
    
    rag = HTMLRAG()
    rag.process_html(html_content)
    
    # Test cases
    test_queries = [
        "How do I sign up?",
        "I want to search for something",
        "How to go back to main page",
        "Can I get free shipping?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        relevant_nodes = rag.find_relevant_nodes(query)
        for rank, result in enumerate(relevant_nodes, 1):
            node = result.node
            print(f"{rank}. Score: {result.score:.3f} ({result.match_type})")
            print(f"   Node: {node.tag} - '{node.text_content}'")
            print(f"   Context: {rag.get_context_for_node(node.backend_node_id)}")

if __name__ == "__main__":
    test_html_rag()