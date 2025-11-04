from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class IMDBRAGImpl:

    IMDBvectordb = None;
    def __init__(self, data):
        self.data = data

    def createIMDBDocuments(self):
        loader = DataFrameLoader(self.data, page_content_column="Movie Description")
        IMDBdocuments = loader.load()
        return IMDBdocuments

    def splitIMDBData(self, documents):
        IMDBTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap  = 100,
            is_separator_regex=False,
            length_function = len
        )
        IMDBsplit  = IMDBTextSplitter.split_documents(documents)
        return IMDBsplit
    
    def createVectorStore(self, splitData):
        """Create vector store using FREE HuggingFace embeddings"""
        print("  → Loading embedding model (first time may download model)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.IMDBvectordb = FAISS.from_documents(splitData, embeddings)
        print("  → Vector store created!")
        
    
    def retrieveIMDBData(self, query):
        IMDBretriever = self.IMDBvectordb.as_retriever(search_type="similarity", search_kwargs={"k":3})
        IMDBresults = IMDBretriever.invoke(query)  # Updated to use invoke instead of get_relevant_documents
        return IMDBresults
    
    def formatIMDBResults(self, IMDBresults):
        formattedResults = ""
        for doc in IMDBresults:
            formattedResults += doc.page_content + "\n\n"
        return formattedResults
    
    def llmchain(self):
        """Setup LLM - tries Ollama (free) first, then falls back to OpenAI"""
        try:
            from langchain_community.llms import Ollama
            from langchain_core.prompts import PromptTemplate
            
            llm = Ollama(model="llama3.2:latest", temperature=0)
            
            template = """You are a helpful movie assistant. Use the context to answer the question. 
            Context:{context} Question: {input} Answer:"""
            
            prompt = PromptTemplate(template=template, input_variables=["context", "input"])
            chain = prompt | llm
            return chain
            
        except Exception as e:
            print(f"⚠️ Ollama not available, trying OpenAI: {e}")
    
    def generateIMDBAnswer(self, query, IMDBresults):
        chain = self.llmchain()
        context = self.formatIMDBResults(IMDBresults)
        IMDBanswer = chain.invoke({"input": query, "context": context})
        
        # Extract text from response
        if isinstance(IMDBanswer, dict):
            return IMDBanswer.get('text', str(IMDBanswer))
        return str(IMDBanswer)
    
    def processIMDBQuery(self, query):
        IMDBresults = self.retrieveIMDBData(query)
        IMDBanswer = self.generateIMDBAnswer(query, IMDBresults)
        return IMDBanswer

    
