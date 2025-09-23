from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH
import os

logger = get_logger(__name__)

def load_vector_store():
    try:
        logger.info("Creating vector store...")
        embedding_model = get_embedding_model()

        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"Loading existing vector store from {DB_FAISS_PATH}")
            return FAISS.load_local(
                DB_FAISS_PATH, 
                embedding_model,
                allow_dangerous_deserialization=True
                )
        else:
            logger.warning(f"Vector store path {DB_FAISS_PATH} does not exist") 
    
        logger.info("Vector store created successfully.")

    except Exception as e:
        error_message = f"Error creating vector store: {str(e)}"
        logger.error(error_message)
    
    #creating new vector store function
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to save in vector store.")
        logger.info(f"Gnerating a new vector store {DB_FAISS_PATH}")

        embedding_model = get_embedding_model()
        db = FAISS.from_documents(text_chunks, embedding_model) 
        db.save_local(DB_FAISS_PATH) #saves the vector store in the specified path
        
        logger.info("Vector store saved successfully.")
        return db
        
    except Exception as e:
        error_message = f"Failed to create a new vector store: {str(e)}"
        logger.error(error_message)
