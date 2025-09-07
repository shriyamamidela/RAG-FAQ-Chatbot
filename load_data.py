"""
FAQ Data Loading and Chunking Module

This module handles loading FAQ data from text files and splitting it into chunks
suitable for vector storage and retrieval.
"""

import os
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def load_faq_data(file_path: str) -> str:
    """
    Load FAQ data from a text file.
    
    Args:
        file_path (str): Path to the FAQ text file
        
    Returns:
        str: Raw FAQ text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"FAQ file not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading FAQ data: {str(e)}")


def parse_faq_sections(text: str) -> List[Dict[str, str]]:
    """
    Parse FAQ text into structured Q&A pairs.
    
    Args:
        text (str): Raw FAQ text
        
    Returns:
        List[Dict[str, str]]: List of dictionaries with 'question', 'answer', and 'category' keys
    """
    faqs = []
    
    # Split by sections (marked by ##)
    sections = re.split(r'\n## ', text)
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        category = lines[0].replace('#', '').strip()
        
        # Find Q&A pairs within each section
        current_q = None
        current_a = []
        
        for line in lines[1:]:
            line = line.strip()
            if line.startswith('Q:'):
                # Save previous Q&A if exists
                if current_q and current_a:
                    faqs.append({
                        'question': current_q,
                        'answer': ' '.join(current_a),
                        'category': category
                    })
                
                # Start new question
                current_q = line[2:].strip()
                current_a = []
            elif line.startswith('A:'):
                current_a.append(line[2:].strip())
            elif current_a and line:  # Continue answer if we're in an answer block
                current_a.append(line)
        
        # Add the last Q&A pair
        if current_q and current_a:
            faqs.append({
                'question': current_q,
                'answer': ' '.join(current_a),
                'category': category
            })
    
    return faqs


def create_documents_from_faqs(faqs: List[Dict[str, str]]) -> List[Document]:
    """
    Convert FAQ data into LangChain Document objects.
    
    Args:
        faqs (List[Dict[str, str]]): List of FAQ dictionaries
        
    Returns:
        List[Document]: List of LangChain Document objects
    """
    documents = []
    
    for faq in faqs:
        # Create a comprehensive text that includes question, answer, and category
        content = f"Category: {faq['category']}\n\nQuestion: {faq['question']}\n\nAnswer: {faq['answer']}"
        
        document = Document(
            page_content=content,
            metadata={
                'question': faq['question'],
                'answer': faq['answer'],
                'category': faq['category'],
                'source': 'faqs.txt'
            }
        )
        documents.append(document)
    
    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents (List[Document]): List of LangChain Document objects
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[Document]: List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


def load_and_process_faqs(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Complete pipeline to load, parse, and chunk FAQ data.
    
    Args:
        file_path (str): Path to the FAQ text file
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[Document]: List of processed and chunked Document objects
    """
    # Load raw text
    raw_text = load_faq_data(file_path)
    
    # Parse into structured Q&A pairs
    faqs = parse_faq_sections(raw_text)
    
    # Convert to Document objects
    documents = create_documents_from_faqs(faqs)
    
    # Chunk the documents
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    
    return chunks


if __name__ == "__main__":
    # Test the module
    try:
        faq_file = "faqs.txt"
        if os.path.exists(faq_file):
            chunks = load_and_process_faqs(faq_file)
            print(f"Successfully processed {len(chunks)} FAQ chunks")
            print(f"Sample chunk: {chunks[0].page_content[:200]}...")
        else:
            print(f"FAQ file '{faq_file}' not found. Please create it first.")
    except Exception as e:
        print(f"Error: {str(e)}")
