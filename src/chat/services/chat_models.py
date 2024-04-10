from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_chat0 = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.001)
openai_chat1 = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.25)
gemini = ChatGoogleGenerativeAI(model='gemini-pro',google_api_key=os.getenv('GEMINI_API_KEY'))
chat_models = {'openai0':openai_chat0,'openai1':openai_chat1,'google':gemini}
