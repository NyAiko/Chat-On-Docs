from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,UnstructuredCSVLoader,UnstructuredExcelLoader,UnstructuredHTMLLoader,UnstructuredPowerPointLoader,UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.vectordb.services.qdrant_setup import insert_docs
from typing import List
import asyncio

def split_docs(documents:list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=75)
    splitted_documents = text_splitter.split_documents(documents)
    return splitted_documents

def load_pdf(filepath:str):
    loader = PyPDFLoader(filepath)
    return loader.load()

def load_docx(filepath:str):
    loader = Docx2txtLoader(filepath)
    return loader.load()

def load_word(filepath:str):
    loader = UnstructuredWordDocumentLoader(filepath)
    return loader.load()

async def load_docs_to_vdb(collection_id:str, filepath:str,extension):
    extension_to_loader = {
        'pdf':load_pdf,
        'docx':load_docx,
        'doc':load_word,
    }

    if extension==None:
        file_extension = filepath.split('.')[-1].lower()
    else:
        file_extension = extension
    
    if file_extension in extension_to_loader:
        loaded_doc = extension_to_loader[file_extension](filepath)
        loaded_doc = split_docs(loaded_doc)

    else:
        raise ValueError('Unsupported file extension')
    await insert_docs(collection_name=collection_id, docs=loaded_doc)

async def load_files_to_vdb(collection_id:str, filepaths:List[str],extension):
    print('Loading Files to VDB')
    #tasks = [load_docs_to_vdb(collection_id, filepath,extension=extension) for filepath in filepaths]
    #await asyncio.gather(*tasks)
    for filepath in filepaths:
        await load_docs_to_vdb(collection_id, filepath,extension=extension)
    
    

### Collection_id: From the s3 folder name
### Filepath: s3 file_path
### Filepath: 