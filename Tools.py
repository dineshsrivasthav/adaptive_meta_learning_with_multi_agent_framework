import os
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from RAG_workflow import get_or_create_vector_db, get_ensemble_retriever
from typing import Optional
#Web tools
#ArXiv, ##Google Scholar, Semantic Scholar,
# Exa Search, , ##Tavily, SerpAPI
# Youtube transcripts, ##News, ##Social media, Trends

os.environ["SERPER_API_KEY"] = ''
os.environ["SERPAPI_API_KEY"] = ''

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

search = GoogleSerperAPIWrapper()

# SerpAPI
serper_tool = Tool(
  name="Intermediate Answer",
  func=search.run,
  description="Useful for search-based queries. Fetch information from the web.",
)

#Arxiv
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

arxiv_wrapper = ArxivAPIWrapper(top_k_results=25,doc_content_chars_max=3000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
arxiv_tool = Tool(
  name="Arxiv",
  func=arxiv.invoke,
  description="Useful for querying from Arxiv research papers to understand any research directions and advancements.",
)
#arxiv.invoke(query)

#Semantic Scholar
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
semantic_scholar = SemanticScholarQueryRun()
#SemanticScholarQueryRun().invoke('evolution in meta learning strategies')
semantic_scholar_tool = Tool(
  name="Semantic scholar",
  func=semantic_scholar.invoke,
  description="Useful for querying from Semantic scholar research papers to understand any research directions and advancements.",
)

#Duckduckgo
search_toolt = DuckDuckGoSearchRun()
search_tool = Tool(
  name="Duckduckgo_websearch",
  func=search_toolt,
  description="Useful for querying from Duckduckgo browser. Fetch relevant hits from the web.",
)

#Trends
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
ttool = GoogleTrendsQueryRun(api_wrapper=GoogleTrendsAPIWrapper())
trends_tool = Tool(
  name="Google Trends",
  func=ttool.run,
  description="Useful for querying Google trends data to understand the global search trends, and hot topics.",
)

#Exa
from exa_py import Exa
from langchain_core.tools import tool

exa = Exa(api_key="")


@tool
def search_and_contents_f(query: str):
    """Search for webpages based on the query and retrieve their contents."""
    # This combines two API endpoints: search and contents retrieval
    return exa.search_and_contents(
        query, use_autoprompt=True, num_results=5, text=True, highlights=True
    )
search_and_contents = Tool(
  name="search_and_contents",
  func=search_and_contents_f,
  description="Search for webpages based on the query and retrieve their contents",
)

@tool
def find_similar_and_contents_f(url: str):
    """Search for webpages similar to a given URL and retrieve their contents.
    The url passed in should be a URL returned from `search_and_contents`.
    """
    # This combines two API endpoints: find similar and contents retrieval
    return exa.find_similar_and_contents(url, num_results=5, text=True, highlights=True)
find_similar_and_contents = Tool(
  name="find_similar_and_contents",
  func=find_similar_and_contents_f,
  description="Search for webpages similar to a given URL and retrieve their contents. The url passed in should be a URL returned from 'search_and_contents",
)


# Vector DB Ensemble Retriever Tool
def get_vector_db_retriever_tool(vector_db_path: str = "./chroma_db", 
                                 use_rerank: bool = False,
                                 k: int = 5):
    """
    Create a tool that uses the ensemble retriever from the vector DB.
    
    Args:
        vector_db_path: Path to the persistent vector DB
        use_rerank: Whether to use Cohere reranking
        k: Number of documents to retrieve
    
    Returns:
        Tool instance for vector DB retrieval
    """
    # Initialize vector DB and retriever
    db, embeddings = get_or_create_vector_db(
        persist_directory=vector_db_path,
        pdf_folder=None
    )
    
    ensemble_retriever = get_ensemble_retriever(
        db, 
        doc_splits=None,  
        use_rerank=use_rerank,
        k=k
    )
    
    def vector_db_search(query: str) -> str:
        """
        Search the vector database for relevant information about deepfake detection,
        attack patterns, research papers, and related topics.
        
        Args:
            query: Search query string
        
        Returns:
            Retrieved documents as formatted string
        """
        try:
            docs = ensemble_retriever.get_relevant_documents(query)
            if not docs:
                return "No relevant documents found in the vector database."
            
            result = f"Found {len(docs)} relevant documents:\n\n"
            for i, doc in enumerate(docs, 1):
                result += f"Document {i}:\n"
                result += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                result += f"Content: {doc.page_content[:500]}...\n\n"
            
            return result
        except Exception as e:
            return f"Error searching vector database: {str(e)}"
    
    vector_db_tool = Tool(
        name="vector_database_search",
        func=vector_db_search,
        description=(
            "Search the curated vector database for information about deepfake detection, "
            "attack patterns, research papers, datasets, and related topics. "
            "This database contains information from local documents and web-crawled content. "
            "Use this tool when you need to find specific information that might be in the "
            "curated knowledge base."
        )
    )
    
    return vector_db_tool


# Initialize vector DB tool (can be configured via environment variables)
VECTOR_DB_PATH = os.environ.get('VECTOR_DB_PATH', './chroma_db')
USE_RERANK = os.environ.get('USE_RERANK', 'false').lower() == 'true'
vector_db_tool = get_vector_db_retriever_tool(
    vector_db_path=VECTOR_DB_PATH,
    use_rerank=USE_RERANK,
    k=5
)

tools=[search_tool, scrape_tool, arxiv_tool, semantic_scholar_tool, serper_tool, 
       trends_tool, search_and_contents, find_similar_and_contents, vector_db_tool]


