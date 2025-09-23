from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.llm import load_llm_model
from app.components.vector_store import load_vector_store

from app.config.config import HUGGNINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException


logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """
Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Contxt: {context}

Question: {question}

Answer: 
"""
def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        
        db = load_vector_store()

        if not db:
            raise CustomException("Vector store is not loaded properly.")
        
        llm = load_llm_model()
        
        if not llm:
            raise CustomException("LLM model is not loaded properly.")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_type="similarity", search_kwargs={"k":1}), #k=1 most relevant data, if u increase k then more context related will show *generally k=3
            return_source_documents=False, #False to not return source documents in the response coz chatbot only needs answer
            chain_type_kwargs={"prompt": set_custom_prompt()}
        )

        logger.info("RetrievalQA chain created successfully.")

        return qa_chain

    except Exception as e:
        error_message = f"Error creating QA chain: {str(e)}"
        logger.error(error_message)
        raise CustomException(error_message)

    
 