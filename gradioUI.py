"""
IMDB RAG ChatBot - Gradio Web Interface
Launch a web UI for querying movies using RAG
"""

import gradio as gr
import os
from IMDBDataSetup import IMDBDataSetup
from IMDBRAGImpl import IMDBRAGImpl

# Global variable to store RAG instance
rag_instance = None
setup_complete = False
setup_status = ""

def initialize_rag():
    """Initialize the RAG system once at startup"""
    global rag_instance, setup_complete, setup_status
    
    if setup_complete:
        return "✅ System already initialized!"
    
    try:
        status_msgs = []
        
        # Step 1: Load data
        status_msgs.append("📂 Loading IMDB dataset...")
        csv_file = "IMDb_Dataset.csv"
        
        if not os.path.exists(csv_file):
            return f"❌ Error: '{csv_file}' not found! Please add your CSV file to the project folder."
        
        IMDBDataSetup.importData(csv_file)
        status_msgs.append("✅ Data loaded successfully")
        
        # Step 2: Clean data
        status_msgs.append("🧹 Cleaning data...")
        IMDBDataSetup.cleanData()
        status_msgs.append("✅ Data cleaned")
        
        # Step 3: Generate descriptions
        status_msgs.append("📝 Generating movie descriptions...")
        IMDBDataSetup.generateMovieDescriptions()
        status_msgs.append(f"✅ Generated descriptions for {len(IMDBDataSetup.IMDBDataFrame)} movies")
        
        # Step 4: Create RAG pipeline
        status_msgs.append("🔧 Creating RAG pipeline...")
        rag_instance = IMDBRAGImpl(IMDBDataSetup.IMDBDataFrame)
        
        status_msgs.append("📚 Creating documents...")
        documents = rag_instance.createIMDBDocuments()
        status_msgs.append(f"✅ Created {len(documents)} documents")
        
        status_msgs.append("✂️ Splitting documents into chunks...")
        split_data = rag_instance.splitIMDBData(documents)
        status_msgs.append(f"✅ Created {len(split_data)} text chunks")
        
        status_msgs.append("🧠 Creating vector store (this may take a moment)...")
        rag_instance.createVectorStore(split_data)
        status_msgs.append("✅ Vector store created!")
        
        setup_complete = True
        setup_status = "\n".join(status_msgs) + "\n\n🎉 IMDB RAG ChatBot is Ready!"
        
        return setup_status
        
    except Exception as e:
        return f"❌ Error during initialization: {str(e)}"


def search_movies(query, search_only=False):
    """
    Search for movies and optionally generate answers
    
    Args:
        query: User's question/search term
        search_only: If True, only return search results without LLM answer
    """
    global rag_instance, setup_complete
    
    if not setup_complete or rag_instance is None:
        return "⚠️ System not initialized! Click 'Initialize System' first."
    
    if not query or not query.strip():
        return "⚠️ Please enter a question or search term."
    
    try:
        # Retrieve relevant movies
        results = rag_instance.retrieveIMDBData(query)
        
        # Format the retrieved documents
        response = f"🔍 **Search Results for:** '{query}'\n\n"
        response += "=" * 60 + "\n\n"
        
        for i, doc in enumerate(results, 1):
            response += f"**Result {i}:**\n{doc.page_content}\n\n"
            response += "-" * 60 + "\n\n"
        
        # If search_only mode, return just the results
        if search_only:
            return response
        
        # Otherwise, generate answer using LLM
        try:
            response += "🤖 **AI-Generated Answer:**\n\n"
            answer = rag_instance.generateIMDBAnswer(query, results)
            response += answer
        except Exception as e:
            response += f"⚠️ Could not generate AI answer: {str(e)}\n"
            response += "💡 Tip: Make sure Ollama is running or set OPENAI_API_KEY"
        
        return response
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def chat_interface(message, history):
    """Chatbot interface for Gradio"""
    if not setup_complete:
        return "⚠️ Please initialize the system first using the 'Setup' tab!"
    
    try:
        # Get answer with LLM
        results = rag_instance.retrieveIMDBData(message)
        answer = rag_instance.generateIMDBAnswer(message, results)
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}\n\n💡 Make sure Ollama is running or set your OPENAI_API_KEY"


def get_sample_movies():
    """Get some sample movies from the dataset"""
    if not setup_complete:
        return "⚠️ Initialize the system first!"
    
    try:
        df = IMDBDataSetup.IMDBDataFrame
        sample = df[['Title', 'Director', 'Genre', 'IMDb Rating', 'Year']].head(10)
        return sample.to_markdown(index=False)
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio Interface
with gr.Blocks(title="🎬 IMDB RAG ChatBot", theme=gr.themes.Soft()) as app:
    
    gr.Markdown(
        """
        # 🎬 IMDB Movie ChatBot with RAG
        
        Ask questions about movies and get AI-powered answers based on the IMDB dataset!
        
        **Powered by:** LangChain + FAISS + HuggingFace Embeddings + Ollama/OpenAI
        """
    )
    
    with gr.Tabs():
        
        # Tab 1: Setup
        with gr.Tab("🔧 Setup"):
            gr.Markdown("### Initialize the RAG System")
            gr.Markdown("Click the button below to load data and create the vector database.")
            
            init_button = gr.Button("🚀 Initialize System", variant="primary", size="lg")
            init_output = gr.Textbox(label="Setup Status", lines=15, max_lines=20)
            
            init_button.click(fn=initialize_rag, outputs=init_output)
        
        # Tab 2: Chat Interface
        with gr.Tab("💬 Chat"):
            gr.Markdown("### Interactive Movie Assistant")
            gr.Markdown("Ask questions in natural language and get AI-generated answers!")
            
            chatbot = gr.ChatInterface(
                fn=chat_interface,
                examples=[
                    "What are some good action movies?",
                    "Tell me about The Godfather",
                    "Recommend high-rated comedy films",
                    "What movies are directed by Christopher Nolan?",
                    "Find thriller movies from the 2000s"
                ],
                title="",
                description=""
            )
        
        # Tab 3: Search Only
        with gr.Tab("🔍 Search"):
            gr.Markdown("### Vector Search (No LLM)")
            gr.Markdown("Search for relevant movies without generating an AI answer (faster!).")
            
            with gr.Row():
                search_input = gr.Textbox(
                    label="Search Query", 
                    placeholder="e.g., action movies, Christopher Nolan, comedy films...",
                    lines=1
                )
            
            search_button = gr.Button("🔍 Search", variant="primary")
            search_output = gr.Markdown(label="Search Results")
            
            search_button.click(
                fn=lambda q: search_movies(q, search_only=True),
                inputs=search_input,
                outputs=search_output
            )
            
            # Example searches
            gr.Examples(
                examples=[
                    ["action movies"],
                    ["Christopher Nolan"],
                    ["high rated movies"],
                    ["comedy films"],
                    ["sci-fi movies"]
                ],
                inputs=search_input
            )
        
        # Tab 4: Q&A with Context
        with gr.Tab("📝 Q&A"):
            gr.Markdown("### Question & Answer with Full Context")
            gr.Markdown("Get both the retrieved context AND the AI-generated answer.")
            
            with gr.Row():
                qa_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about movies...",
                    lines=2
                )
            
            qa_button = gr.Button("💡 Get Answer", variant="primary")
            qa_output = gr.Markdown(label="Answer with Context")
            
            qa_button.click(
                fn=lambda q: search_movies(q, search_only=False),
                inputs=qa_input,
                outputs=qa_output
            )
        
        # Tab 5: Dataset Info
        with gr.Tab("📊 Dataset Info"):
            gr.Markdown("### Dataset Statistics")
            
            info_button = gr.Button("📊 Show Sample Movies")
            info_output = gr.Markdown(label="Sample Data")
            
            info_button.click(fn=get_sample_movies, outputs=info_output)
            
            gr.Markdown(
                """
                ---
                ### About This Dataset
                - **Source:** IMDB Movie Dataset
                - **Columns:** Title, Director, Genre, Star Cast, Duration, Rating, Year
                - **Total Movies:** 2,762
                
                ### RAG Configuration
                - **Embedding Model:** all-MiniLM-L6-v2 (HuggingFace)
                - **Vector Store:** FAISS
                - **Chunk Size:** 500 characters
                - **Chunk Overlap:** 100 characters
                - **Retrieval:** Top 3 similar documents
                
                ### LLM Options
                - **Primary:** Ollama (llama3.2 - Free, Local)
                - **Fallback:** OpenAI GPT-3.5-turbo (Requires API key)
                """
            )
    
    gr.Markdown(
        """
        ---
        ### 💡 Tips:
        - **First Time?** Go to the Setup tab and initialize the system
        - **Fast Search?** Use the Search tab (no LLM needed)
        - **Best Answers?** Use the Chat or Q&A tab (requires LLM)
        - **No Ollama?** Set `OPENAI_API_KEY` environment variable
        
        ### 🛠️ Requirements:
        - Ollama installed with llama3.2 model, OR
        - OpenAI API key set in environment
        """
    )


if __name__ == "__main__":
    print("🚀 Starting IMDB RAG ChatBot...")
    print("📡 Server will start on http://127.0.0.1:7860")
    print("🌐 Opening in browser...")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True to create a public link
        inbrowser=True  # Automatically open in browser
    )
