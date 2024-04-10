from src.chat.services.prompts import summarization_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
llm = ChatOpenAI(model = 'gpt-3.5-turbo', temperature=0)

summarize = summarization_prompt |llm | StrOutputParser()
