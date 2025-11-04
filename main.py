import os
from IMDBDataSetup import IMDBDataSetup
from IMDBRAGImpl import IMDBRAGImpl

def main():
    # Step 1: Set up OpenAI API key (you need to set this)
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Step 2: Import and clean data
    print("Step 1: Importing data...")
    csv_file = "IMDb_Dataset.csv"  # Replace with your actual CSV file path
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        print("Please provide the path to your IMDB CSV file.")
        return
    
    IMDBDataSetup.importData(csv_file)
    
    print("\nStep 2: Cleaning data...")
    IMDBDataSetup.cleanData()
    
    print("\nStep 3: Generating movie descriptions...")
    IMDBDataSetup.generateMovieDescriptions()
    
    print(f"\nTotal movies in dataset: {len(IMDBDataSetup.IMDBDataFrame)}")
    print("\nSample data:")
    print(IMDBDataSetup.IMDBDataFrame[['Title', 'Movie Description']].head(2))

    # Step 4: Create RAG implementation
    print("\nStep 4: Setting up RAG implementation...") 
    ragImpl = IMDBRAGImpl(IMDBDataSetup.IMDBDataFrame)
    
    print("Creating documents...")
    documents = ragImpl.createIMDBDocuments()
    print(f"Created {len(documents)} documents")
    
    print("Splitting documents into chunks...")
    splitData = ragImpl.splitIMDBData(documents)
    print(f"Created {len(splitData)} text chunks")
    
    print("Creating vector store (this may take a moment)...")
    ragImpl.createVectorStore(splitData)
    print("Vector store created successfully!")
    
    # Step 5: Interactive query loop
    print("\n" + "="*60)
    print("🎬 IMDB RAG ChatBot is Ready!")
    print("="*60)
    print("\nYou can ask questions like:")
    print("  - What are some good action movies?")
    print("  - Tell me about comedy films")
    print("  - Recommend movies with high ratings")
    print("  - What movies are directed by [director name]?")
    print("\nType 'quit' to exit\n")
    
    while True:
        query = input("Your question: ")
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue
        
        try:
            print("\n🔍 Searching and generating answer...\n")
            answer = ragImpl.processIMDBQuery(query)
            print(f"💬 Answer: {answer}\n")
            print("-" * 60)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure Ollama is running with llama3.2 model installed")


if __name__ == "__main__":
    main()
