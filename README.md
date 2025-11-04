# IMDB RAG ChatBot - Implementation Guide

## ✅ What You Have Now

Your IMDB RAG chatbot is **fully implemented** with the following components:

### Files Created:
1. **`IMDBDataSetup.py`** - Loads and cleans CSV data
2. **`IMDBRAGImpl.py`** - Complete RAG implementation with FREE embeddings
3. **`main.py`** - Interactive chatbot interface
4. **`test_rag.py`** - Test script to verify vector search works
5. **`requirements.txt`** - All dependencies

---

## 🎯 How RAG Works in Your Implementation

### The RAG Pipeline:

```
User Query → Vector Search → Retrieve Relevant Docs → LLM Generates Answer
```

### Step-by-Step Process:

1. **Data Loading** (`IMDBDataSetup`)
   - Loads `IMDb_Dataset.csv`
   - Cleans data (removes NaN, duplicates)
   - Creates descriptive text for each movie

2. **Document Creation** (`createIMDBDocuments`)
   - Converts each movie into a Document object
   - Uses "Movie Description" column as searchable content

3. **Text Chunking** (`splitIMDBData`)
   - Breaks text into 500-character chunks
   - 100-character overlap for context continuity
   - Total: 2762 chunks from your dataset

4. **Vector Embedding** (`createVectorStore`)
   - Uses **HuggingFace "all-MiniLM-L6-v2"** (FREE, no API key)
   - Converts text to numerical vectors
   - Stores in FAISS vector database

5. **Semantic Search** (`retrieveIMDBData`)
   - Takes user query
   - Finds 3 most similar movie descriptions
   - Uses cosine similarity in vector space

6. **Answer Generation** (`generateIMDBAnswer`)
   - Formats retrieved context
   - Sends to LLM with prompt
   - Returns natural language answer

---

## 🚀 Running Your RAG Chatbot

### Option 1: Without LLM (Just Vector Search)

Test that retrieval works:

```powershell
python test_rag.py
```

This shows you the top 3 matching movies for any query without needing an LLM.

### Option 2: With Ollama (FREE, Local LLM)

**Setup Ollama:**
1. Download from https://ollama.ai/download
2. Install Ollama
3. Open PowerShell and run:
   ```powershell
   ollama pull llama3.2:latest
   ```
4. Keep Ollama running (it runs as a service)

**Run the chatbot:**
```powershell
python main.py
```
---

## 📊 Example Queries

Once running, try these:

### Basic Queries:
- "What are some good action movies?"
- "Tell me about The Godfather"
- "Recommend comedy films"
- "What movies have high ratings?"

### Advanced Queries:
- "Find movies directed by Christopher Nolan"
- "What are the best thriller movies from 2020?"
- "Tell me about sci-fi movies with Tom Hanks"
- "What are some family-friendly adventure movies?"

---

## 🔧 How Each Component Works

### 1. Vector Embeddings (The Magic)

```python
# Text gets converted to numbers:
"The Movie 'Inception'..." → [0.234, -0.891, 0.445, ...]

# Queries also become vectors:
"action movies" → [0.198, -0.823, 0.512, ...]

# Similar movies have similar vectors (close in vector space)
```

### 2. FAISS Vector Store

- **F**acebook **A**I **S**imilarity **S**earch
- Extremely fast similarity search
- Works on CPU (no GPU needed)
- Indexes all 2762 movie chunks

### 3. Retrieval Process

```python
Query: "action movies"
  ↓
Vector: [0.198, -0.823, ...]
  ↓
FAISS finds closest vectors
  ↓
Top 3 matches:
  1. "The Movie 'Mad Max: Fury Road'... action"
  2. "The Movie 'John Wick'... action" 3. "The Movie 'Die Hard'... action"
```

### 4. LLM Integration

The LLM receives:
```
Context:
The Movie 'Mad Max: Fury Road', directed by George Miller...
The Movie 'John Wick', directed by Chad Stahelski...
The Movie 'Die Hard', directed by John McTiernan...

Question: What are some good action movies?

Answer: [LLM generates response using the context]
```

---

## 🛠️ Customization Options

### Adjust Chunk Size
In `IMDBRAGImpl.py`, modify:
```python
chunk_size = 500  # Decrease for more precise chunks
chunk_overlap = 100  # Increase for more context
```

### Change Number of Results
```python
search_kwargs={"k": 3}  # Change 3 to 5 or more
```

### Use Different Embedding Model
```python
model_name="all-MiniLM-L6-v2"  # Try "all-mpnet-base-v2" for better quality
```

### Add Filters
Modify `retrieveIMDBData` to filter by year, genre, rating, etc.

---

## 📈 Performance Notes

### First Run:
- Downloads embedding model (~90MB) - one time only
- Takes ~30 seconds to create vector store
- Subsequent runs use cached model

### Query Speed:
- Vector search: <100ms
- LLM generation: 1-5 seconds (Ollama) or 0.5-2 seconds (OpenAI)

### Memory Usage:
- Embedding model: ~500MB RAM
- Vector store: ~50MB RAM
- LLM (Ollama): ~4GB RAM

---

## 🐛 Troubleshooting

### "No module named 'sentence_transformers'"
```powershell
pip install sentence-transformers
```

### "Ollama connection error"
- Make sure Ollama is installed
- Run `ollama pull llama3.2:latest`
- Check if Ollama service is running

### "OpenAI API key not set"
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

### Slow performance
- First run downloads models (one-time)
- Try reducing dataset size for testing
- Use GPU if available (change device='cpu' to device='cuda')

### "Out of memory"
- Reduce chunk_size to 300
- Process fewer documents at once
- Close other applications

---

## 🎓 Understanding the Code

### Key Classes:

**`IMDBDataSetup`**
- Static class for data management
- Stores DataFrame as class variable
- Generates movie descriptions

**`IMDBRAGImpl`**
- Instance-based RAG implementation
- Stores vector database instance
- Handles entire RAG pipeline

### Key Methods:

**`createIMDBDocuments()`**
- Input: DataFrame
- Output: List of Document objects
- Purpose: Convert structured data to searchable documents

**`splitIMDBData()`**
- Input: Documents
- Output: Chunked documents
- Purpose: Break long text for better search

**`createVectorStore()`**
- Input: Chunked documents
- Output: FAISS vector database
- Purpose: Enable semantic search

**`retrieveIMDBData()`**
- Input: User query (string)
- Output: Top 3 relevant documents
- Purpose: Find matching movies

**`generateIMDBAnswer()`**
- Input: Query + retrieved docs
- Output: Natural language answer
- Purpose: Generate human-readable response

---

## 🚦 Next Steps

### Beginner:
1. Run `test_rag.py` to see vector search working
2. Install Ollama and try `main.py`
3. Experiment with different queries

### Intermediate:
1. Modify chunk_size and k parameters
2. Add genre/year filters to queries
3. Try different embedding models
4. Save/load vector store to disk

### Advanced:
1. Implement conversation history
2. Add multi-query retrieval
3. Implement re-ranking
4. Add hybrid search (keyword + semantic)
5. Create web interface with Streamlit/Flask

---

## 📚 Learn More

### RAG Concepts:
- **Embeddings**: Convert text to numbers that capture meaning
- **Vector Search**: Find similar items using mathematical distance
- **Chunking**: Split text to improve retrieval precision
- **Context Window**: Amount of text LLM can process

### Langchain Components:
- `Document`: Container for text + metadata
- `TextSplitter`: Breaks documents into chunks
- `VectorStore`: Database for vector search
- `Retriever`: Interface for searching documents
- `Chain`: Connects components together

---

## ✨ Your Implementation Advantages

1. **✅ FREE embeddings** - No API costs for vector search
2. **✅ Offline capable** - Works with Ollama (no internet for LLM)
3. **✅ Fast search** - FAISS is optimized for speed
4. **✅ Scalable** - Can handle thousands of movies
5. **✅ Flexible** - Easy to swap LLMs or embedding models
6. **✅ Production-ready** - Proper error handling and fallbacks

---

## 🎉 Congratulations!

You now have a fully functional RAG chatbot that:
- Loads and processes IMDB data
- Creates semantic search index
- Retrieves relevant movies
- Generates natural language answers
- Works with free or paid LLMs

**Your chatbot is ready to use!** 🚀
