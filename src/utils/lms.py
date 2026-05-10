"""
Factory for creating language models. This allows us to easily switch between different LLM providers and models 
by changing the configuration in one place.
Function for each provider that takes in the model name and other parameters and returns an instance of the corresponding LLM class.
"""

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI

from dotenv import load_dotenv
import os
load_dotenv()


def deepseek_model(model_name: str, temperature: float):
    """Returns an instance of the ChatDeepSeek model with the specified parameters."""
    return ChatDeepSeek(
        model=model_name,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    
def lmstudio_model(model_name: str, temperature: float):
    """Returns an instance of the ChatOpenAI model configured for LM Studio with the specified parameters."""    
    return ChatOpenAI(
        model= model_name,
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        temperature=temperature,)
    
def groq_model(model_name: str, temperature: float):
    """Returns an instance of the ChatGroq model with the specified parameters."""
    return ChatGroq(
        model= model_name,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    
def ollama_model(model_name: str, temperature: float):
    """Returns an instance of the ChatOpenAI model configured for Ollama with the specified parameters."""
    return ChatOpenAI(
        model= model_name,
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        temperature=temperature,)
    
def openrouter_model(model_name: str, temperature: float):
    """Returns an instance of the ChatOpenAI model configured for OpenRouter with the specified parameters."""
    return ChatOpenAI(
        model= model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=temperature,)
    
def cloudflare_model(model_name: str, temperature: float):
    """Returns an instance of the ChatCloudflareWorkersAI model with the specified parameters."""
    return ChatCloudflareWorkersAI(
        model_name= model_name,
        temperature=temperature,)
    

# Dictionary mapping model names to their corresponding factory functions. This allows us to easily retrieve the desired model by name.
models = {
    "deepseek": lambda tmp : deepseek_model("deepseek-chat", tmp),
    "granite4-micro": lambda tmp  : lmstudio_model("granite-4.0-micro", tmp),
    "granite4-micro-or": lambda tmp  : openrouter_model("ibm-granite/granite-4.0-h-micro", tmp),
    "granite4-micro-cf": lambda tmp  : cloudflare_model("@cf/ibm-granite/granite-4.0-h-micro", tmp),
    "granite4-tiny" : lambda tmp  : lmstudio_model("ibm/granite-4-h-tiny", tmp),
    "qwen-32b": lambda tmp : groq_model("qwen/qwen3-32b", tmp),
    "qwen-32b-or": lambda tmp : openrouter_model("qwen/qwen3-32b", tmp),    
    "oss-20b": lambda tmp : groq_model("openai/gpt-oss-20b", tmp),
    "oss-120b": lambda tmp : groq_model("openai/gpt-oss-120b", tmp), 
    "oss-120b-ol": lambda tmp : ollama_model("gpt-oss:120b-cloud", tmp),        
}

def get_model(model_name: str, temperature :float = 0.0):
    """Retrieves the specified model with the given temperature. Raises an error if the model name is not found."""
    if model_name in models:
        return models[model_name](temperature)
    else:
        raise ValueError(f"Model '{model_name}' not found.")