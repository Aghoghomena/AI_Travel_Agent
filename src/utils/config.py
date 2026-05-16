# config.py
import dspy
from dotenv import load_dotenv

load_dotenv()

from src.utils.lms import get_model
llm = get_model("deepseek", temperature=0.7)

# DSPy LM — same DeepSeek model, picked up via LiteLLM (reads DEEPSEEK_API_KEY)
dspy.configure(lm=dspy.LM("deepseek/deepseek-chat", temperature=0.0))