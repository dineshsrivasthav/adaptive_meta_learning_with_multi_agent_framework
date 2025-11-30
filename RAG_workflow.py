import os
import fitz 
from tqdm import tqdm
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from typing import List, Optional


# Default paths
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_PDF_FOLDER = './local_deepfake_docs/'


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def load_pdfs(pdf_folder):
    """Load all PDFs from a folder and return as documents."""
    documents = []
    if not os.path.exists(pdf_folder):
        print(f"Warning: PDF folder {pdf_folder} does not exist. Skipping PDF loading.")
        return documents
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            text = extract_text_from_pdf(pdf_path)
            text = text.replace('\t', ' ').replace('\n', ' ').replace('-\n', '').replace('- ','')
            documents.append({"page_content": text, "metadata": {"source": pdf_file}})
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    return documents


def get_or_create_vector_db(persist_directory: str = DEFAULT_PERSIST_DIR, 
                           pdf_folder: Optional[str] = None,
                           embeddings_model: str = "nomic-ai/nomic-embed-text-v1"):
    """
    Get existing vector DB or create a new one from PDFs.
    
    Args:
        persist_directory: Directory to persist the vector DB
        pdf_folder: Folder containing PDFs to load (if creating new DB)
        embeddings_model: Model name for embeddings
    
    Returns:
        Chroma vectorstore instance and embeddings
    """
    embeddings = SentenceTransformerEmbeddings(
        model_name=embeddings_model, 
        model_kwargs={"trust_remote_code": True}
    )
    
    # Check if vector DB already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vector DB from {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print(f"Creating new vector DB at {persist_directory}")
        if pdf_folder:
            documents = load_pdfs(pdf_folder)
            if documents:
                document_objects = [Document(page_content=doc['page_content'], 
                                           metadata=doc['metadata']) 
                                 for doc in documents]
                
                text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type='percentile')
                doc_splits = text_splitter.split_documents(document_objects)
                
                db = Chroma.from_documents(
                    doc_splits, 
                    embedding=embeddings, 
                    persist_directory=persist_directory
                )
            else:
                # Create empty DB
                db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        else:
            # Create empty DB
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    return db, embeddings


def add_documents_to_vector_db(db, documents: List[Document], embeddings):
    """
    Add new documents to an existing vector DB.
    
    Args:
        db: Chroma vectorstore instance
        documents: List of Document objects to add
        embeddings: Embeddings model
    """
    if not documents:
        return
    
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type='percentile')
    doc_splits = text_splitter.split_documents(documents)
    
    # Add documents to existing collection
    db.add_documents(doc_splits)
    print(f"Added {len(doc_splits)} document chunks to vector DB")


def get_ensemble_retriever(db, doc_splits: Optional[List[Document]] = None, 
                          use_rerank: bool = False,
                          k: int = 5):
    """
    Create an ensemble retriever from the vector DB.
    
    Args:
        db: Chroma vectorstore instance
        doc_splits: Document splits for BM25 (if None, will use vector retriever only)
        use_rerank: Whether to use Cohere reranking
        k: Number of documents to retrieve
    
    Returns:
        Ensemble retriever (with optional reranking)
    """
    vectorstore_retriever = db.as_retriever(search_kwargs={"k": k})
    
    
    if doc_splits is None:
        ensemble_retriever = vectorstore_retriever
    else:
        keyword_retriever = BM25Retriever.from_documents(doc_splits)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vectorstore_retriever, keyword_retriever],
            weights=[0.7, 0.3]
        )
    
    if use_rerank:
        cohere_api_key = os.environ.get('COHERE_API_KEY', '')
        if not cohere_api_key:
            print("Warning: COHERE_API_KEY not set. Skipping reranking.")
            return ensemble_retriever
        
        compressor = CohereRerank(model="rerank-english-v3.0")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble_retriever
        )
        return compression_retriever
    
    return ensemble_retriever


# Main execution for standalone use
if __name__ == "__main__":
    pdf_folder = os.environ.get('PDF_FOLDER', DEFAULT_PDF_FOLDER)
    persist_dir = os.environ.get('VECTOR_DB_PATH', DEFAULT_PERSIST_DIR)
    
    db, embeddings = get_or_create_vector_db(
        persist_directory=persist_dir,
        pdf_folder=pdf_folder
    )
    
    documents = load_pdfs(pdf_folder)
    doc_splits = None
    if documents:
        document_objects = [Document(page_content=doc['page_content'], 
                                   metadata=doc['metadata']) 
                         for doc in documents]
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type='percentile')
        doc_splits = text_splitter.split_documents(document_objects)
    
    # Create ensemble retriever
    ensemble_retriever = get_ensemble_retriever(db, doc_splits=doc_splits, use_rerank=False)
    print("Vector DB and ensemble retriever ready")
