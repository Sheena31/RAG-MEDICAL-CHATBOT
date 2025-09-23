from langchain_huggingface import HuggingFaceEndpoint
from config.config import HF_TOKEN, HUGGNINGFACE_REPO_ID

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm_model(huggingface_repo_id: str = HUGGNINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info("Loading LLM model from Hugging Face Hub...")

        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=hf_token,
            temperature=0.3, #randomness of the model *lower the temprature lower creative the model
            max_new_tokens=256, #maximum length of the response
            return_full_text=False, #return only the generated text
        )

        logger.info("LLM model loaded successfully.")

        return llm

    except Exception as e:
        error_message = f"Error loading LLM model: {str(e)}"
        logger.error(error_message)
        raise CustomException(error_message)

