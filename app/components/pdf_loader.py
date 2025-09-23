import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter #splitting text into chunks

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)
def load_pdf_file():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"The specified data path '{DATA_PATH}' does not exist.")
        logger.info(f"Loading PDF files from directory: {DATA_PATH}")

        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader) #loading all pdf files from data folder
        documents = loader.load() #stores all pdf files in documents variable

        if not documents:
            raise CustomException(f"No PDF files found in the specified directory '{DATA_PATH}'.")
        else:
            logger.info(f"Successfully loaded {len(documents)} PDF files.")

        return documents
    
    except Exception as e:
        error_message = CustomException("Failed to load PDF" , e)
        logger.error(str(error_message))
        return []
    
    

def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents provided for text chunking.")
        
        logger.info(f"Splitting {len(documents)} documents into chunks.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Created {len(text_chunks)} text chunks.")

        return text_chunks

    except Exception as e:
        error_message = CustomException("Failed to generate chunks" , e)
        logger.error(str(error_message))
        return []




