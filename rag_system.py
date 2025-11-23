import pandas as pd
import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# Configuration
PROJECT_ID = "your-google-cloud-project-id"
LOCATION = "us-central1"
DATA_FILE_PATH = "../data/medquad.csv"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

class MedicalRAGSystem:
    def __init__(self, data_path):
        """
        Initializes the RAG system by loading the data and models.
        """
        self.data_path = data_path
        self.df = None
        self.vector_db = [] # Stores embedding vectors
        self.docs_metadata = [] # Stores text content
        
        # Load Models
        # 'text-embedding-004' is optimized for semantic retrieval
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        # Use the base model or your tuned model name here
        self.generative_model = GenerativeModel("gemini-1.5-flash-002")
        
        self._load_and_index_data()

    def _get_embedding(self, text):
        """Generates vector embeddings for a given text string."""
        if not text or not isinstance(text, str):
            return np.zeros(768)
        
        # Vertex AI embedding generation
        embeddings = self.embedding_model.get_embeddings([text])
        return embeddings[0].values

    def _load_and_index_data(self):
        """
        Loads the CSV and creates in-memory vector embeddings for retrieval.
        Note: For production, use a Vector Database like Vertex AI Vector Search or Pinecone.
        """
        print("Loading dataset and generating embeddings (Indexing)...")
        self.df = pd.read_csv(self.data_path).dropna().head(100) # Limiting to 100 for demo speed
        
        for _, row in self.df.iterrows():
            # Combine question and answer to enrich semantic context
            content = f"Question: {row['question']}\nAnswer: {row['answer']}"
            
            vector = self._get_embedding(content)
            
            if np.any(vector):
                self.vector_db.append(vector)
                self.docs_metadata.append(content)
        
        print(f"Indexing complete. {len(self.vector_db)} documents indexed.")

    def query(self, user_query):
        """
        Full RAG Pipeline:
        1. Embed user query.
        2. Retrieve most relevant document.
        3. Generate answer using LLM + Context.
        """
        # 1. Vectorize Query
        query_vec = self._get_embedding(user_query)
        
        # 2. Similarity Search (Cosine Similarity / Dot Product)
        scores = np.dot(self.vector_db, query_vec)
        best_idx = np.argmax(scores)
        retrieved_context = self.docs_metadata[best_idx]
        
        print(f"\n[INFO] Retrieved Context:\n{retrieved_context[:150]}...\n")
        
        # 3. Augmented Generation
        prompt = f"""
        You are a specialized medical assistant. Use the context provided below to answer the user's question.
        If the answer is not in the context, state that you do not have that information.
        
        CONTEXT:
        {retrieved_context}
        
        USER QUESTION:
        {user_query}
        """
        
        response = self.generative_model.generate_content(prompt)
        return response.text

# --- Main Execution ---
if __name__ == "__main__":
    # Instantiate the system
    rag_system = MedicalRAGSystem(DATA_FILE_PATH)
    
    # Example Loop
    while True:
        user_input = input("\nEnter a medical question (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
            
        answer = rag_system.query(user_input)
        print(f"\nModel Response:\n{answer}")