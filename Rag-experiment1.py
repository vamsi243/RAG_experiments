from io import BytesIO
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain

chroma_client = chromadb.Client()
#collection = chroma_client.create_collection(name="my_collection")
# Initialize OpenAI API (replace with your API key)
#import open AI api key.
import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

import pickle
os.environ['OPENAI_API_KEY']="API key"

# Streamlit app layout
st.title("RAG Assistant")

# Document upload
# Document upload
uploaded_file = st.file_uploader("Upload a PDF document")
#print(uploaded_file)

if uploaded_file is not None:
    # Get the local file path of the uploaded file
    file_path = uploaded_file.name
    #print(file_path)

# Prompt input field
prompt = st.text_input("Enter your prompt:")
#print(prompt)
if prompt and uploaded_file:
    pdf_reader = PdfReader("Dsa.pdf")

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    #print(text)
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
    chunks = text_splitter.split_text(text=text)
    #print(chunks[1])
 # # embeddings
    store_name = uploaded_file.name[:-4]
    st.write(f'{store_name}')
        # st.write(chunks)
    embeddings = OpenAIEmbeddings()
    VectorStore = Chroma.from_texts(chunks, embeddings)
    persist_directory = "chroma_db"

    vectordb = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_directory)

    vectordb.persist()
    #faiss_index = FAISS.from_texts(chunks, OpenAIEmbeddings())
    docs = vectordb.similarity_search(prompt, k=2)
    #print(docs[1])
    model = ChatOpenAI(temperature=0)
    chain = load_qa_chain(model, chain_type="stuff",verbose=True)

    query = prompt
    matching_docs = vectordb.similarity_search(query)
    print("matching docs000", matching_docs)
    answer =  chain.run(input_documents=matching_docs, question=query)
    print(answer)
    st.write("Response:", answer)
