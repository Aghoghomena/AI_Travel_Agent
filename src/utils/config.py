# config.py
from dotenv import load_dotenv

load_dotenv()

from src.utils.lms import get_model
llm = get_model("deepseek", temperature=0.7)