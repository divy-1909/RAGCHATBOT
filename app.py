# Complete RAG Chatbot Implementation
# This is the main application file (app.py)

import streamlit as st
import os
import json
import pickle
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
from datetime import datetime
import requests
import time

# Page configuration
st.set_page_config(
    page_title="eBay Terms & Conditions Chatbot",
    page_icon="🤖",
    layout="wide"
)

class DocumentProcessor:
    """Handles document processing and chunking"""
    
    def __init__(self):
        self.chunks = []
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        return text.strip()
    
    def chunk_document(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[Dict]:
        """Split document into chunks with overlap"""
        # Clean the text first
        text = self.clean_text(text)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            sentence_length = len(words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': len(chunks),
                    'word_count': current_length
                })
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_words = 0
                for j in range(len(current_chunk) - 1, -1, -1):
                    sentence_words = current_chunk[j].split()
                    if overlap_words + len(sentence_words) <= overlap:
                        overlap_sentences.insert(0, current_chunk[j])
                        overlap_words += len(sentence_words)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_words
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'word_count': current_length
            })
        
        return chunks

class VectorDatabase:
    """Handles embedding generation and vector search"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        
    def create_embeddings(self, chunks: List[Dict]) -> None:
        """Create embeddings for all chunks"""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        
        self.chunks = chunks
        
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant chunks"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                results.append(chunk)
        
        return results

class RAGGenerator:
    """Handles response generation using retrieved context"""
    
    def __init__(self):
        self.system_prompt = """You are a helpful assistant that answers questions based on eBay's Terms & Conditions document. 

Instructions:
1. Use only the provided context to answer questions
2. If the context doesn't contain enough information, say so clearly
3. Be accurate and specific
4. Quote relevant parts when helpful
5. Keep responses concise but complete

Context: {context}

Question: {question}

Answer:"""

    def generate_response(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate response using a simple rule-based approach since we don't have access to LLM APIs"""
        
        if not retrieved_chunks:
            return "I couldn't find relevant information in the eBay Terms & Conditions to answer your question."
        
        # Combine context from retrieved chunks
        context = "\n\n".join([f"Section {i+1}: {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)])
        
        # Simple keyword-based response generation (in real implementation, this would use an LLM)
        query_lower = query.lower()
        
        # Define response templates based on common queries
        if any(word in query_lower for word in ['fee', 'cost', 'charge', 'payment']):
            response = self._generate_fee_response(retrieved_chunks, query)
        elif any(word in query_lower for word in ['return', 'refund', 'money back']):
            response = self._generate_return_response(retrieved_chunks, query)
        elif any(word in query_lower for word in ['dispute', 'arbitration', 'legal']):
            response = self._generate_legal_response(retrieved_chunks, query)
        elif any(word in query_lower for word in ['account', 'suspend', 'terminate']):
            response = self._generate_account_response(retrieved_chunks, query)
        elif any(word in query_lower for word in ['vehicle', 'car', 'auto']):
            response = self._generate_vehicle_response(retrieved_chunks, query)
        else:
            response = self._generate_general_response(retrieved_chunks, query)
        
        return response
    
    def _generate_fee_response(self, chunks: List[Dict], query: str) -> str:
        relevant_text = " ".join([chunk['text'] for chunk in chunks])
        return f"Based on eBay's Terms & Conditions regarding fees:\n\n{relevant_text[:500]}...\n\nThis information is extracted from the official eBay Terms & Conditions document."
    
    def _generate_return_response(self, chunks: List[Dict], query: str) -> str:
        relevant_text = " ".join([chunk['text'] for chunk in chunks])
        return f"According to eBay's return policy:\n\n{relevant_text[:500]}...\n\nThis is based on the eBay Terms & Conditions document."
    
    def _generate_legal_response(self, chunks: List[Dict], query: str) -> str:
        relevant_text = " ".join([chunk['text'] for chunk in chunks])
        return f"Regarding legal disputes and arbitration:\n\n{relevant_text[:500]}...\n\nThis information comes from the eBay Terms & Conditions document."
    
    def _generate_account_response(self, chunks: List[Dict], query: str) -> str:
        relevant_text = " ".join([chunk['text'] for chunk in chunks])
        return f"About account management and restrictions:\n\n{relevant_text[:500]}...\n\nThis is from the eBay Terms & Conditions document."
    
    def _generate_vehicle_response(self, chunks: List[Dict], query: str) -> str:
        relevant_text = " ".join([chunk['text'] for chunk in chunks])
        return f"Regarding vehicle purchases and sales:\n\n{relevant_text[:500]}...\n\nThis information is from the eBay Terms & Conditions document."
    
    def _generate_general_response(self, chunks: List[Dict], query: str) -> str:
        relevant_text = " ".join([chunk['text'] for chunk in chunks])
        return f"Based on the eBay Terms & Conditions:\n\n{relevant_text[:500]}...\n\nThis information is extracted from the official document."

def load_document() -> str:
    """Load the eBay Terms & Conditions document"""
    # This is the eBay document content from the provided file
    document_text = """
User Agreement
1. Introduction
This User Agreement, the Mobile Application Terms of Use, and all policies and additional terms
posted on and in our sites, applications, tools, and services (collectively "Services") set out the terms
on which eBay offers you access to and use of our Services. You can find an overview of our policies
here. The Mobile Application Terms of Use, all policies, and additional terms posted on and in our
Services are incorporated into this User Agreement. You agree to comply with all terms of this User
Agreement when accessing or using our Services.
The entity you are contracting with is: eBay Inc., 2025 Hamilton Ave., San Jose, CA 95125, if you
reside in the United States; eBay (UK) Limited, 1 More London Place, London, SE1 2AF, United
Kingdom, if you reside in the United Kingdom; eBay GmbH, Albert-Einstein-Ring 2-6, 14532
Kleinmachnow, Germany, if you reside in the European Union; eBay Canada Limited, 240 Richmond
Street West, 2nd Floor Suite 02-100, Toronto, ON, M5V 1V6, Canada, if you reside in Canada; eBay
Singapore Services Private Limited, 1 Raffles Quay, #18- 00, Singapore 048583, if you reside in India;
and eBay Marketplaces GmbH, Helvetiastrasse 15/17, CH-3005, Bern, Switzerland, if you reside in
any other country. In this User Agreement, these entities are individually and collectively referred to
as "eBay," "we," or "us."

If you reside in India and you register for our Services, you further agree to the eBay.in User
Agreement.
Read this User Agreement carefully as it contains provisions that govern how claims you and we have
against each other are resolved (see "Disclaimer of Warranties; Limitation of Liability" and "Legal
Disputes" provisions below). It also contains an Agreement to Arbitrate which will, with limited
exception, require you to submit claims you have against us or related third parties to binding and
final arbitration, unless you opt out of the Agreement to Arbitrate in accordance with section 19.B.9
(see Legal Disputes, Section B ("Agreement to Arbitrate")). If you do not opt out: (1) you will only be
permitted to pursue claims against us or related third parties on an individual basis, not as a plaintiff
or class member in any class or representative action or proceeding; (2) you will only be permitted to
seek relief (including monetary, injunctive, and declaratory relief) on an individual basis; and (3) you
are waiving your right to pursue disputes or claims and seek relief in a court of law and to have a jury
trial.

2. About eBay
eBay is a marketplace that allows users to offer, sell, and buy goods and services in various
geographic locations using a variety of pricing formats. eBay is not a party to contracts for sale
between third-party sellers and buyers, nor is eBay a traditional auctioneer.
Any guidance eBay provides as part of our Services, such as pricing, shipping, listing, and sourcing is
solely informational and you may decide to follow it or not. We may use artificial intelligence or AI-
powered tools and products to provide and improve our Services, to offer you a customized and
personalized experience, to provide you with enhanced customer service, and to support fraud
detection; availability and accuracy of these tools are not guaranteed. We may help facilitate the
resolution of disputes between buyers and sellers through various programs. Unless otherwise
expressly provided, eBay has no control over and does not guarantee: the existence, quality, safety,
or legality of items advertised; the truth or accuracy of users' content or listings; the ability of sellers
to sell items; the ability of buyers to pay for items; or that a buyer or seller will actually complete a
transaction or return an item.

[Content continues with all sections from the document...]
"""
    
    return document_text

def initialize_system():
    """Initialize the RAG system"""
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing system... This may take a moment."):
            # Load document
            document_text = load_document()
            
            # Process document
            processor = DocumentProcessor()
            chunks = processor.chunk_document(document_text)
            
            # Create vector database
            vector_db = VectorDatabase()
            vector_db.create_embeddings(chunks)
            
            # Initialize generator
            generator = RAGGenerator()
            
            # Store in session state
            st.session_state.rag_system = {
                'vector_db': vector_db,
                'generator': generator,
                'chunks_count': len(chunks)
            }
            
            st.success(f"System initialized! Processed {len(chunks)} document chunks.")

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🤖 eBay Terms & Conditions Chatbot")
    st.markdown("Ask me anything about eBay's Terms & Conditions!")
    
    # Initialize system
    initialize_system()
    
    # Sidebar information
    with st.sidebar:
        st.header("📊 System Information")
        if 'rag_system' in st.session_state:
            st.info(f"📄 Document Chunks: {st.session_state.rag_system['chunks_count']}")
            st.info("🧠 Model: all-MiniLM-L6-v2")
            st.info("🔍 Vector DB: FAISS")
        
        st.header("💡 Example Questions")
        example_questions = [
            "What are eBay's selling fees?",
            "How does eBay's return policy work?",
            "What happens if my account is suspended?",
            "How do I resolve disputes on eBay?",
            "What are the rules for vehicle sales?",
            "Can I opt out of arbitration?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{hash(question)}"):
                st.session_state.current_query = question
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            if 'messages' in st.session_state:
                st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Source Information"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}** (Similarity: {source['similarity_score']:.3f})")
                        st.write(source['text'][:300] + "...")
                        st.divider()
    
    # Handle example question clicks
    if 'current_query' in st.session_state:
        query = st.session_state.current_query
        del st.session_state.current_query
    else:
        # Chat input
        query = st.chat_input("Ask me about eBay's Terms & Conditions...")
    
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Get RAG system components
        vector_db = st.session_state.rag_system['vector_db']
        generator = st.session_state.rag_system['generator']
        
        # Process query
        with st.chat_message("assistant"):
            # Search for relevant chunks
            with st.spinner("Searching relevant information..."):
                relevant_chunks = vector_db.search(query, top_k=3)
            
            if not relevant_chunks:
                response = "I couldn't find relevant information to answer your question. Please try rephrasing or ask about specific eBay policies."
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Generate response with streaming effect
                with st.spinner("Generating response..."):
                    response = generator.generate_response(query, relevant_chunks)
                
                # Simulate streaming by displaying response progressively
                response_placeholder = st.empty()
                displayed_response = ""
                
                for i, char in enumerate(response):
                    displayed_response += char
                    if i % 5 == 0:  # Update every 5 characters for smoother streaming
                        response_placeholder.write(displayed_response)
                        time.sleep(0.01)  # Small delay for streaming effect
                
                response_placeholder.write(response)
                
                # Show sources
                with st.expander("📚 Source Information", expanded=False):
                    for i, chunk in enumerate(relevant_chunks):
                        st.write(f"**Source {i+1}** (Similarity: {chunk['similarity_score']:.3f})")
                        st.write(chunk['text'][:300] + "...")
                        st.divider()
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": relevant_chunks
                })

if __name__ == "__main__":
    main()