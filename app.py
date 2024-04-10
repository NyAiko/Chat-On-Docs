import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from tempfile import NamedTemporaryFile
import asyncio
from src.vectordb.services.load_to_vdb import load_files_to_vdb
from src.chat.services.chat import ChatOnDocs

async def load_documents_to_vector_db(collection_id, uploaded_files, corresponding_metadata = {}):
    if 'loaded_files' not in st.session_state:
        st.session_state.loaded_files = []
    
    for file in uploaded_files:
        if file.name not in st.session_state.loaded_files:
            extension = file.name.split('.')[-1]
            with NamedTemporaryFile(delete=False, dir='temp/') as temp_file:
                temp_file.write(file.read())
                filepath = '/'.join(temp_file.name.split('/')[-2:])
                await load_files_to_vdb(collection_id=collection_id, filepaths=[filepath], extension=extension)
                st.session_state.loaded_files.append(file.name)
                st.session_state.file_metadata[file.name] = filepath

def init():
    load_dotenv()
    st.set_page_config(
        page_title="Your own ChatGPT",
        page_icon="🤖"
    )
    
    

async def main():
    init()
    st.header('Chat On your Own documents')
    st.sidebar.header("Chat-On-Docs🤖")
    # Initialize session state variables
    if 'loaded_files' not in st.session_state:
        st.session_state.loaded_files = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'file_metadata' not in st.session_state:
        st.session_state.file_metadata = {}

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
        if uploaded_files:
            await load_documents_to_vector_db(collection_id="ma_collection", 
                                              uploaded_files=uploaded_files)
        selected_docs = st.multiselect("Which Doc you want to chat on?", 
                                       options = [file.name for file in uploaded_files],
                                       help='Please Select')
        print(selected_docs)
        print(st.session_state.file_metadata)

    sources = []

    for s in st.session_state.file_metadata:
        sources.append(st.session_state.file_metadata[s])
    
    upload = len(st.session_state.loaded_files)>0
    has_select_docs = len(sources)>0
    print('Upload *** :', upload)
    print('Selected Docs *** :', has_select_docs)
    chat = ChatOnDocs(collection_id = 'ma_collection', filenames = sources,upload = upload)
    user_input = st.text_input("Your message: ", key="user_input")
    
    if st.button("Send",key='sendmsg'):
        if user_input:
            print(user_input)
            with st.spinner("Thinking..."):
                response = await chat(user_input)
                st.session_state.messages.append(user_input)
                st.session_state.messages.append(response)
        messages = st.session_state.get('messages', [])

        for i, msg in enumerate(reversed(messages)):
            if i%2==False:
                message(msg, is_user=False, key=str(i) + '_ai')
            else:
                message(msg, is_user=True, key=str(i) + '_user')
if __name__ == '__main__':
    asyncio.run(main())
    
