from src.chat.services.prompts import summarization_prompt
from langchain_core.output_parsers import StrOutputParser
from src.chat.services.chat_models import chat_models

llm = chat_models['google']
summarize = summarization_prompt |llm | StrOutputParser()
