"""
Vector Store Management Module

This module handles creating and managing FAISS vector stores for FAQ retrieval.
"""

import os
import pickle
from typing import List, Optional
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


class FAQVectorStore:
    """
    A class to manage FAISS vector store for FAQ retrieval.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the FAQ Vector Store.
        
        Args:
            embedding_model (str): Name of the embedding model to use
        """
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
        self.index_path = "faq_index"
        self.pkl_path = "faq_index.pkl"
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create a FAISS vector store from documents.
        
        Args:
            documents (List[Document]): List of LangChain Document objects
            
        Returns:
            FAISS: Created vector store
        """
        print(f"Creating vector store from {len(documents)} documents...")
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        print(f"Vector store created successfully with {len(documents)} documents")
        return self.vectorstore
    
    def save_vectorstore(self, path: Optional[str] = None) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path (str, optional): Path to save the vector store. Defaults to self.index_path
        """
        if self.vectorstore is None:
            raise ValueError("No vector store to save. Create one first.")
        
        save_path = path or self.index_path
        self.vectorstore.save_local(save_path)
        print(f"Vector store saved to {save_path}")
    
    def load_vectorstore(self, path: Optional[str] = None) -> FAISS:
        """
        Load a vector store from disk.
        
        Args:
            path (str, optional): Path to load the vector store from. Defaults to self.index_path
            
        Returns:
            FAISS: Loaded vector store
        """
        load_path = path or self.index_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        self.vectorstore = FAISS.load_local(
            load_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"Vector store loaded successfully from {load_path}")
        return self.vectorstore
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents in the vector store.
        
        Args:
            query (str): Search query
            k (int): Number of similar documents to return
            
        Returns:
            List[Document]: List of similar documents
        """
        if self.vectorstore is None:
            raise ValueError("No vector store loaded. Load or create one first.")
        
        # Perform similarity search
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
    
    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query (str): Search query
            k (int): Number of similar documents to return
            
        Returns:
            List[tuple]: List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("No vector store loaded. Load or create one first.")
        
        # Perform similarity search with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        return docs_with_scores
    
    def get_vectorstore_info(self) -> dict:
        """
        Get information about the current vector store.
        
        Returns:
            dict: Dictionary containing vector store information
        """
        if self.vectorstore is None:
            return {"status": "No vector store loaded"}
        
        # Get basic information
        info = {
            "status": "Vector store loaded",
            "embedding_model": self.embedding_model,
            "index_type": "FAISS"
        }
        
        # Try to get document count
        try:
            if hasattr(self.vectorstore, 'index') and hasattr(self.vectorstore.index, 'ntotal'):
                info["document_count"] = self.vectorstore.index.ntotal
        except:
            info["document_count"] = "Unknown"
        
        return info


def create_faq_vectorstore(documents: List[Document], 
                          embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                          save_path: str = "faq_index") -> FAQVectorStore:
    """
    Create and save a FAQ vector store.
    
    Args:
        documents (List[Document]): List of LangChain Document objects
        embedding_model (str): Name of the embedding model to use
        save_path (str): Path to save the vector store
        
    Returns:
        FAQVectorStore: Created vector store manager
    """
    # Create vector store manager
    vectorstore_manager = FAQVectorStore(embedding_model)
    
    # Create vector store from documents
    vectorstore_manager.create_vectorstore(documents)
    
    # Save the vector store
    vectorstore_manager.save_vectorstore(save_path)
    
    return vectorstore_manager


def load_faq_vectorstore(load_path: str = "faq_index",
                        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAQVectorStore:
    """
    Load an existing FAQ vector store.
    
    Args:
        load_path (str): Path to load the vector store from
        embedding_model (str): Name of the embedding model to use
        
    Returns:
        FAQVectorStore: Loaded vector store manager
    """
    # Create vector store manager
    vectorstore_manager = FAQVectorStore(embedding_model)
    
    # Load the vector store
    vectorstore_manager.load_vectorstore(load_path)
    
    return vectorstore_manager


if __name__ == "__main__":
    # Test the module
    try:
        from load_data import load_and_process_faqs
        
        # Load FAQ data
        faq_file = "faqs.txt"
        if os.path.exists(faq_file):
            documents = load_and_process_faqs(faq_file)
            
            # Create vector store
            vectorstore_manager = create_faq_vectorstore(documents)
            
            # Test search
            test_query = "What is the return policy?"
            results = vectorstore_manager.search_similar_documents(test_query, k=3)
            
            print(f"\nTest search for: '{test_query}'")
            print(f"Found {len(results)} similar documents:")
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. {doc.metadata.get('question', 'No question')}")
                print(f"   Answer: {doc.page_content[:100]}...")
            
            # Get vector store info
            info = vectorstore_manager.get_vectorstore_info()
            print(f"\nVector store info: {info}")
            
        else:
            print(f"FAQ file '{faq_file}' not found. Please create it first.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
