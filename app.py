import streamlit as st
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
import tempfile
import os

######################################
if "doc_available" not in st.session_state:
    st.session_state.doc_available = False

if "param" not in st.session_state:
    st.session_state.param = True

## Generate Docs from PDF File
def generate_docs(uploaded_file):
    pdf_bytes = uploaded_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    os.unlink(tmp_file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, add_start_index=True, separators='\n'
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits,True
    
## Adding Embeddings to the Vector Base
def add_vector_database(all_splits,vector_store):
    _ = vector_store.add_documents(documents=all_splits) 

## Get answers based on a context and query
@st.cache_data
def search(query,_vector_store,_model,k=3):
    prompt = hub.pull("rlm/rag-prompt")
    retrieved_docs = _vector_store.similarity_search(query,k=k)
    docs_content =  "\n\n".join(doc.page_content for doc in retrieved_docs)
    llm_chain = prompt | _model
    response = llm_chain.invoke({'question':query,'context':docs_content})
    return response

## START
st.title("PDF file analyzer 🔍")

if "messages" not in st.session_state:
    st.session_state.messages = []

## Enter the API key and upload the document to be analyzed
with st.sidebar:
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload a file", type=None)


if uploaded_file is None:
    st.session_state.param = True
    if st.session_state.doc_available == False:
        st.warning("Set the key and upload a file")
        st.stop()

if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar.")
    st.stop()


try:
    model = init_chat_model(model_provider="openai",model="gpt-4o-mini",api_key=api_key)   

except Exception as e:
    print(e)
    st.warning("Your API Key is invalid, please try again.")
    st.stop()

## Create the vector store
if "vector_store" not in st.session_state:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=api_key)
    st.session_state.vector_store = InMemoryVectorStore(embeddings) 

## Process the uploaded PDF
if uploaded_file != None and st.session_state.param:
    file_extension = uploaded_file.name.split(".")[-1].lower() if "." in uploaded_file.name else "No extension"
    if file_extension == 'pdf':
        docs,st.session_state.doc_available = generate_docs(uploaded_file)
        add_vector_database(docs,st.session_state.vector_store)
        st.session_state.param = False
    else:
        st.warning("Please send a PDF file")
        st.stop()


## Initial message
if len(st.session_state.messages) == 0:
    with st.chat_message('ai'):
        st.markdown(f"Hello, I'm ready to help you with whatever you need about your file {uploaded_file.name}")

## Space to enter queries
user_msg = st.chat_input()

## Generate response
if user_msg:
    st.session_state.messages.append(('user',user_msg))
    response = search(user_msg,st.session_state.vector_store,model)
    st.session_state.messages.append(('ai',response.content))

## Show messages
type_msg = None
for role,message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message) 