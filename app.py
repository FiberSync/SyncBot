from st_on_hover_tabs import on_hover_tabs
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

st.set_page_config(layout="wide")

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

with st.sidebar:
    tabs = on_hover_tabs(tabName=['AI Assistant', 'Order Planner'], 
                         iconName=['dashboard', 'money'], default_choice=0)

if tabs =='AI Assistant':
    st.header("FiberSync AI Assistant")
    
    # Set up environment variable for API key
    os.environ["GROQ_API_KEY"] = "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"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Initialize the ChatGroq model
    llm = ChatGroq(
        model="gemma2-9b-it",  # Use a suitable model name
        temperature=0.3,
        max_retries=2,
    )

    # Function to create a prompt
    prompt_template = ChatPromptTemplate.from_messages([
           ("system", """You are a helpful AI assistant working for FiberSync ( Blockchain Based SCM Application ) 
            Answer questions about anything related to the company(Who purchased it), services, or the textile industry specific Question.,
            Your Artificial AI assistant Name is Sync-Bot(FiberSync's AI Assistant Specialized In Textile Process Management)  
            Answer Your question in {language} language style""",
            ),
            ("human", "{question}")
    ])


    # User input
    question = st.text_input("Please enter your question")


    chain = prompt_template | llm

    options = ['English', 'Roman Urdu(English Urdu Mix)']
    selected_option = st.selectbox('Select Response Language:', options)

    if question:
        # Generate the prompt messages
        prompt = chain.invoke({"question": question,"language": selected_option})
        with st.expander("View Response"):
            st.markdown(prompt.content)

elif tabs == 'Order Planner':
    st.header("FiberSync Order Planner")

    # Load your dataset
    try:
        with open("dset.txt", "r") as f:
            text = f.read()
    except FileNotFoundError:
        st.error("Dataset file not found! Please ensure 'dset.txt' is available in the working directory.")
        text = ""

    if text:
        # Split the text into chunks
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
        chunks = splitter.split_text(text)

        # Convert chunks to Document objects
        from langchain.schema import Document
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Create embeddings
        from langchain.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create a FAISS vector store
        from langchain.vectorstores import FAISS
        vector_db = FAISS.from_documents(documents, embeddings)

        # Define LLM
        llm = ChatGroq(
            model="llama3-70b-8192",  # Use a suitable model name
            temperature=0.3,
            max_retries=2,
        )

        # Define prompt template for combining documents
        from langchain.prompts import PromptTemplate
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the context below to answer the question as best as possible:
            Context: {context}
            Question: {question}
            Answer:""",
        )

        # Define the CombineDocsChain
        from langchain.chains import LLMChain
        combine_docs_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Define the ConversationalRetrievalChain
        from langchain.chains import ConversationalRetrievalChain
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(),
            return_source_documents=False,  # Set to True if source documents are required
            verbose=True,
        )

        # User input for order specifications
        order_spec = st.text_input("Please enter your order specifications")

        if order_spec:
            with st.spinner("Generating order plan..."):
                response = retrieval_chain({"question": order_spec, "chat_history": []})  # Provide chat_history if needed
                with st.expander("View Order Plan"):
                    st.markdown(response["answer"])
    else:
        st.info("Upload or load a valid dataset to proceed.")