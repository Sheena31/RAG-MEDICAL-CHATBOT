from langchain_huggingface import HuggingFaceEmbeddings

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Intializing our hugginface embedding model")

        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #embedding model

        logger.info("Huggingface embedding model loaded successfully")

        return model
    except Exception as e:
        error_message = f"Error loading Huggingface embedding model: {str(e)}"
        logger.error(error_message)
        raise error_message