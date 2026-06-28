import os
import logging
from typing import Optional, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)

# Verify which .env file is being loaded
env_file = find_dotenv()
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f".env file found at: {env_file}")

# Load the env file and check if it executed
env_loaded = load_dotenv(env_file) if env_file else load_dotenv()
logger.info(f"load_dotenv() executed: {env_loaded}")

# Log keys for verification
google_api_key = os.environ.get("GOOGLE_API_KEY")
gemini_api_key = os.environ.get("GEMINI_API_KEY")

logger.info(f"GOOGLE_API_KEY exists? {google_api_key is not None}")
if google_api_key:
    logger.info(f"GOOGLE_API_KEY length: {len(google_api_key)}")
    logger.info(f"GOOGLE_API_KEY first 4 chars: {google_api_key[:4]}")

logger.info(f"GEMINI_API_KEY exists? {gemini_api_key is not None}")
if gemini_api_key:
    logger.info(f"GEMINI_API_KEY length: {len(gemini_api_key)}")
    logger.info(f"GEMINI_API_KEY first 4 chars: {gemini_api_key[:4]}")

# If only GEMINI_API_KEY exists, mirror it into GOOGLE_API_KEY
if gemini_api_key and not google_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    logger.info("Mirrored GEMINI_API_KEY into GOOGLE_API_KEY")

def get_llm() -> Optional[BaseChatModel]:
    """
    Returns the configured LLM instance based on priority: Gemini -> OpenRouter -> Ollama.
    """
    # Natively reads GOOGLE_API_KEY
    if os.environ.get("GOOGLE_API_KEY"):
        logger.info("Using Gemini LLM (gemini-2.5-flash). Reading GOOGLE_API_KEY from env.")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        
    if os.environ.get("OPENROUTER_API_KEY"):
        logger.info("Using OpenRouter LLM.")
        return ChatOpenAI(
            model="meta-llama/llama-3-8b-instruct:free", # Cost-effective OpenRouter fallback
            temperature=0.1,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        
    if os.environ.get("OLLAMA_HOST") and ChatOllama:
        logger.info("Using Ollama LLM.")
        return ChatOllama(model="llama3", temperature=0.1, base_url=os.environ.get("OLLAMA_HOST"))
        
    logger.warning("No LLM provider configured.")
    return None

def get_llm_status() -> Dict[str, Any]:
    """
    Returns the status of the configured LLM.
    """
    provider = "None"
    model = "None"
    
    if os.environ.get("GOOGLE_API_KEY"):
        provider = "Gemini"
        model = "gemini-2.5-flash"
    elif os.environ.get("OPENROUTER_API_KEY"):
        provider = "OpenRouter"
        model = "meta-llama/llama-3-8b-instruct:free"
    elif os.environ.get("OLLAMA_HOST") and ChatOllama:
        provider = "Ollama"
        model = "llama3"
        
    return {
        "provider": provider,
        "model": model,
        "env_loaded": env_loaded,
        "google_api_key_present": os.environ.get("GOOGLE_API_KEY") is not None,
        "gemini_api_key_present": os.environ.get("GEMINI_API_KEY") is not None,
        # generation_working will be populated at the application level
    }
