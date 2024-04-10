import asyncio
from langchain_openai import ChatOpenAI
from typing import List
from src.chat.services.prompts import rag_prompt, intent_clf, general_chain_prompt,language_detection, input_reformulation
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel
from src.query_vdb.services.find_docs import find_document
from src.summarize.services.summarization import fast_summarization
from src.intent_classifier.services.intent_classifier import classify_intent
from src.chat.services.chat_models import chat_models

chat_models = chat_models

class ChatOnDocs():
    def __init__(self,collection_id:str,filenames: List[str],upload):
        self.collection_id = collection_id
        self.filenames = filenames
        self.upload = upload

    async def detect_language(self, input_query:str):
        input = {'text':input_query}
        chain = language_detection | chat_models['openai0'] | StrOutputParser()
        return chain.invoke(input)

    async def detect_intention(self, input_query:str):
        #chain = intent_clf | ChatOpenAI(model= 'gpt-3.5-turbo', temperature=0.15) |StrOutputParser()
        #input = {'prompt':input_query}
        intent = await classify_intent(input_query)
        return intent

    async def reformulate_prompt(self, input_query:str):
        input = {'question':input_query}
        chain = input_reformulation | chat_models['openai1'] | StrOutputParser()
        return chain.invoke(input)

    async def chat_on_docs(self, input_query:str):
        if (self.upload == True) and (len(self.filenames)>0):
            input = {'question': input_query}#,'language':language}
            retrieve_docs = RunnableLambda(lambda x: asyncio.run(find_document(collection_name=self.collection_id,
                                                                        query_text = x['question'],
                                                                        filenames = self.filenames)))
            process_input = RunnableParallel({'question': RunnableLambda(lambda x: x['question']),
                                            'docs': retrieve_docs})
                                            #'language':RunnableLambda(lambda x: input['language'])})
            
            chain = process_input|rag_prompt|chat_models['google']| StrOutputParser()
            return chain.invoke(input)
        else:
            return """Please upload documents if you haven't yet . Then select which documents would you to chat on ;) """
    async def summarize(self):
        if (self.upload == True) and (len(self.filenames)>0):
            output = await fast_summarization(collection_id = self.collection_id, target_metadata = self.filenames)
            return output
        else:
            return """Please upload documents if you haven't yet . Then select which documents would you to chat on ;) """

    async def casual_chat(self, input_query):#,language=None):
        input = {'prompt':input_query}#,'language':language}
        chain = general_chain_prompt|chat_models['openai1']|StrOutputParser()
        return chain.invoke(input)

    async def chat(self,input_query:str):
        intent = await self.detect_intention(input_query)
        print('User Intent ----', intent)
        #language = await self.detect_language(input_query)
        if 'summary' in intent.lower():
            output= await self.summarize()
        elif 'information retrieval' in intent.lower():
            output = await self.chat_on_docs(input_query)
        else:
            output = await self.casual_chat(input_query)
        return output

    async def __call__(self, input_query:str):
        response = await self.chat(input_query)
        return response