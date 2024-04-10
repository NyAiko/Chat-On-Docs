from langchain_core.prompts import PromptTemplate

general_chain_prompt = PromptTemplate.from_template(
    template = """
                You are an AI assitant. 
                You task is only to perform question answer from documents or document sumamrization.
                Respond to the following:
                {prompt}
                Just answer the question. In case this is outside of your task, answer that you are only here to perform question answering from documents or document summarization
                Your answer: 
                """
)

summarization_prompt = PromptTemplate.from_template(
    template = """
                The following text are the most important text a documents.
                {text}
                Reformulate them to become a concise summary:
                Your summary:
                
            """
)

rag_prompt = PromptTemplate.from_template(
    template="""
                Given the following documents as context
                <documents>
                {docs}
                </documents>
                Answer the following question:
                {question}
                If the answer is not inside of the documents, ask for clarification:
            """
)

intent_clf = PromptTemplate.from_template(
    template = """
                Given the following prompt,
                {prompt}
                Which of the following is the intention behind it?
                Respond only with one of the following intents:
                'asking for summarization'
                'information retrieval'
                'information request'
                'greetings'
                'casual conversation'
                'thanks'
                'complaint'
                Your unique response: 

                """)

input_reformulation = PromptTemplate.from_template(template="""
        Reformulate the following input into English. Try to extract the question from it
        {question}
        Reformulation:
        """)

language_detection = PromptTemplate.from_template(template = """
        In what language is the following text: 
        {text}
        Only respond me with one word.
        """)