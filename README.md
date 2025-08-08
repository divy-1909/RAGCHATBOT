# eBay Terms & Conditions RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about eBay's Terms & Conditions using semantic search and AI-powered response generation.

## 🎯 Project Overview

This chatbot processes eBay's Terms & Conditions document, creates semantic embeddings, and uses a retrieval system to answer user questions with relevant, accurate information from the official document.

## 🚀 Features

- **Semantic Search**: Uses sentence transformers to find relevant document sections
- **Streaming Responses**: Real-time response generation with streaming effect
- **Source Attribution**: Shows which document sections were used to generate answers
- **Interactive UI**: Clean Streamlit interface with example questions
- **Vector Database**: FAISS-powered similarity search
- **Document Processing**: Intelligent chunking with overlap for better context

## 📋 Architecture

```
User Query → Vector Search → Context Retrieval → Response Generation → Streaming Display
     ↓              ↓               ↓                    ↓              ↓
Embedding → FAISS Index → Top-K Chunks → Rule-based Gen → Token Stream
```

### Key Components:

1. **DocumentProcessor**: Cleans and chunks the document into semantic segments
2. **VectorDatabase**: Creates embeddings and handles similarity search using FAISS
3. **RAGGenerator**: Generates contextual responses based on retrieved chunks
4. **Streamlit Interface**: Provides interactive chat interface with streaming

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Text Processing**: NumPy, Regular Expressions
- **Language**: Python 3.8+

## 📁 Project Structure

```
rag-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── data/                 # Document storage (create this folder)
    └── ebay_terms.txt    # eBay Terms & Conditions document
```

## ⚡ Quick Start

### 1. Clone or Download the Project

Create a new folder and save the `app.py` file and `requirements.txt` file.

### 2. Install Python

If you don't have Python installed:
- Go to [python.org](https://python.org/downloads/)
- Download Python 3.8 or newer
- Install it (make sure to check "Add Python to PATH")

### 3. Install Dependencies

Open a terminal/command prompt in your project folder and run:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser automatically!

## 🎮 How to Use

1. **Start the Application**: Run `streamlit run app.py`
2. **Wait for Initialization**: The system will process the document and create embeddings
3. **Ask Questions**: Type questions about eBay's terms in the chat input
4. **View Sources**: Expand the "Source Information" section to see relevant document sections
5. **Try Examples**: Use the sidebar example questions to get started

### Example Questions:

- "What are eBay's selling fees?"
- "How does eBay's return policy work?"
- "What happens if my account is suspended?"
- "How do I resolve disputes on eBay?"
- "What are the rules for vehicle sales?"
- "Can I opt out of arbitration?"

## 🔧 Configuration

### Document Processing Settings:
- **Chunk Size**: 200 words per chunk
- **Overlap**: 50 words between chunks
- **Embedding Model**: all-MiniLM-L6-v2
- **Similarity Threshold**: Cosine similarity
- **Retrieved Chunks**: Top 3 most relevant

### Customization Options:
- Modify chunk size in `DocumentProcessor.chunk_document()`
- Change embedding model in `VectorDatabase.__init__()`
- Adjust number of retrieved chunks in the search function
- Customize response templates in `RAGGenerator`

## 📊 Performance Metrics

- **Document Processing**: ~10,500 words processed into ~50-70 chunks
- **Embedding Generation**: ~2-3 seconds for full document
- **Query Response Time**: ~1-2 seconds per query
- **Memory Usage**: ~200MB for embeddings and index

## 🎯 Sample Queries and Responses

### Query: "What are eBay's selling fees?"

**Response**: Based on eBay's Terms & Conditions regarding fees:

We charge sellers for the use of our Services. In some cases, where buyers receive supplemental Services such as authentication or storage Services for items in certain categories, we may also charge those buyers for such supplemental Services. The fees we charge sellers for using our Services to sell goods and services are listed on our Selling fees pages...

**Sources**: Section 6 (Fees and Taxes)

### Query: "How do I return an item?"

**Response**: According to eBay's return policy:

Most eBay sales go smoothly, but if there's a problem with a purchase covered by eBay Money Back Guarantee, buyers can get their money back if an item didn't arrive, is faulty or damaged, or doesn't match the listing...

**Sources**: Section 14 (Additional Terms - Returns)

## 🚧 Limitations

1. **Simple Response Generation**: Uses rule-based templates instead of advanced LLM
2. **No Real-time Learning**: Doesn't learn from user interactions
3. **Static Document**: Only processes the provided eBay terms document
4. **Limited Context**: Responses are based on retrieved chunks only

## 🔮 Future Enhancements

- **Integration with OpenAI/Hugging Face APIs** for better response generation
- **Multi-document Support** for handling multiple policy documents
- **User Feedback System** for improving response quality
- **Chat History Persistence** across sessions
- **Advanced Filtering** by document sections or topics
- **Export Functionality** for saving conversations

## 🐛 Troubleshooting

### Common Issues:

1. **Installation Errors**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Memory Issues**:
   - Reduce chunk size or use a smaller embedding model
   - Close other applications to free up RAM

3. **Slow Performance**:
   - The first run is slower due to model downloads
   - Subsequent runs should be much faster

4. **Import Errors**:
   ```bash
   pip install --force-reinstall sentence-transformers
   ```

## 📝 Technical Details

### Document Chunking Strategy:
- **Sentence-aware splitting**: Preserves sentence boundaries
- **Overlapping chunks**: 50-word overlap prevents context loss
- **Size optimization**: 200-word chunks balance context and precision

### Embedding Approach:
- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Normalization**: L2 normalization for cosine similarity
- **Index Type**: FAISS IndexFlatIP for inner product similarity

### Response Generation:
- **Context Injection**: Retrieved chunks provide factual basis
- **Template Matching**: Different templates for different query types
- **Source Attribution**: Shows similarity scores and source text

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify Python version is 3.8 or newer
4. Check that you have sufficient system memory (4GB+ recommended)

## 📄 License

This project is for educational and demonstration purposes. The eBay Terms & Conditions document belongs to eBay Inc.

---

**Built for the Amlgo Labs Junior AI Engineer Assignment**

*Demonstrates RAG pipeline implementation, vector databases, semantic search, and interactive AI applications.*