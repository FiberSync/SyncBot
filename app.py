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
from fpdf import FPDF , HTMLMixin
from langchain.prompts import ChatPromptTemplate
import markdown2
from langchain.prompts import PromptTemplate

st.set_page_config(layout="wide", page_title="Sync Tools",page_icon=":gear:")

from fpdf import FPDF


class PDF(FPDF,HTMLMixin):
    def header(self):
        # Rendering logo:
        self.image("https://i.imghippo.com/files/WoXD6395jA.png", 70, 14, 60)
        # Setting font: helvetica bold 15
        self.set_font("Helvetica", style="B", size=12)
        # Moving cursor to the right:
        self.cell(80)
        # Printing title:
        # self.cell(30, 10, "Order Plan", border=1, align="C")
        # Performing a line break:
        self.ln(20)

    def footer(self):
        # Position cursor at 1.5 cm from bottom:
        self.set_y(-15)
        # Setting font: helvetica italic 8
        self.set_font("helvetica", style="I", size=8)
        # Printing page number:
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def create_pdf(content):
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font('Helvetica', size=10)
        html_content = markdown2.markdown(content,extras=["tables"])
        pdf.write_html(html_content)
        return bytes(pdf.output())

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
st.markdown("<style>.stApp{ background-color: #0E1117; }</style>", unsafe_allow_html=True)

with st.sidebar:
    tabs = on_hover_tabs(tabName=['AI Assistant', 'Order Planner','Document Analyzer'],   styles = {'navtab': {'background-color':'#111',
                                                  
                                                  'font-size': '14px',
                                                  'transition': '.3s',
                                                  'white-space': 'nowrap',
                                                  'text-transform': 'capitalize'}},
                         iconName=['sms', 'list_alt','picture_as_pdf'], default_choice=0,)

if tabs =='AI Assistant':
    st.image("https://i.ibb.co/WPSHQmQ/logo-AI.png")
    
    # Set up environment variable for API key
    os.environ["GROQ_API_KEY"] = "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"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Initialize the ChatGroq model
    llm = ChatGroq(
        model="gemma2-9b-it", 
        temperature=0.3,
        max_retries=2,
    )

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

    options = ['English','Spanish','French']
    selected_option = st.selectbox('Select Response Language:', options)

    if question:
        prompt = chain.invoke({"question": question,"language": selected_option})
        with st.expander("View Response"):
            st.markdown(prompt.content)

elif tabs == 'Order Planner':
    st.image("https://i.ibb.co/nB5JfjL/logo-planner-removebg-preview.png")

    try:
        with open("dset.txt", "r") as f:
            text = f.read()
    except FileNotFoundError:
        st.error("Dataset file not found! Please ensure 'dset.txt' is available in the working directory.")
        text = ""

    if text:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
        chunks = splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]


        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create a FAISS vector store
        vector_db = FAISS.from_documents(documents, embeddings)

        # Define LLM
        llm = ChatGroq(
            model="llama3-70b-8192",  # Use a suitable model name
            temperature=0.3,
            max_retries=2,
        )



        prompt_template = PromptTemplate(
            input_variables=["context", "question", "guideline"],
            template="""Use the Order Specifications below to give order plan for textile production company:
            Context: {context}
            Question: {question}
            guideline for output: {guideline}
            Answer:""",
        )

        # Define the CombineDocsChain
        from langchain.chains import LLMChain
        combine_docs_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Define the ConversationalRetrievalChain

        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(),
            return_source_documents=False,  # Set to True if source documents are required
            verbose=True,
            combine_docs_chain_kwargs={'prompt': prompt_template},
        )

        # User input for order specifications
        order_spec = st.text_input("Please Enter Your Order Specifications :")

        if order_spec:
            with st.spinner("Generating order plan..."):
                response = retrieval_chain({"question": order_spec, "guideline": """In output donot include yourself or anyother prompts like
             "Let me know if you have any questions or if you'd like me to generate a new report etc" and also always add paragrpah of methodology in report  answer the question and in professional report style format as your response will direstly 
            be converted into markdown report generator:""", "chat_history": []})
                  # Provide chat_history if needed
                with st.expander("View Order Plan"):
                    st.markdown(response["answer"])

        if order_spec:
            st.download_button(
                    label="Download Report",
                    data=create_pdf(response["answer"]),
                    file_name="plan.pdf",
                    mime="application/pdf",
                    )  
        else:
            st.info("Upload or load a valid dataset to proceed.")
            
elif tabs == 'Document Analyzer':
    st.image("https://i.ibb.co/WPSHQmQ/logo-AI.png")
    st.title("FiberSync Document Analyzer - LLAMA 3.1")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize user input in session state
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Define a toggle for resetting the input field
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    def load_document(file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents

    def setup_vectorstore(documents):
        """Set up a FAISS vector store with document embeddings for retrieval."""
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        doc_chunks = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(doc_chunks, embeddings)
        return vectorstore

    def create_chain(vectorstore):
        llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0
        )
        retriever = vectorstore.as_retriever()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )
        return chain

    def process_user_query(user_input):
        """Process user input and update chat history with assistant response."""
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Processing request..."):
            response = st.session_state.conversation_chain({"question": user_input})
            assistant_response = response["answer"]
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # File upload section
    uploaded_file = st.file_uploader(label="Upload your PDF file", type=["pdf"])


    if uploaded_file:
        file_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load document and set up vectorstore if not already in session state
        if "vectorstore" not in st.session_state:
            documents = load_document(file_path)
            st.session_state.vectorstore = setup_vectorstore(documents)

        # Set up conversation chain if not already in session state
        if "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)


    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        css_class = "user-message" if role == "user" else "assistant-message"
        st.markdown(f"<div class='chat-container {css_class}'>{content}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<div class='input-box-container'>", unsafe_allow_html=True)
    user_input = st.text_input(
        "Ask LLAMA...",
        key=str(st.session_state.input_key),  # Increment key each time to reset input
        label_visibility="collapsed"
    )
    send_button = st.button("Send", key="send_button")


    if send_button and user_input:
        process_user_query(user_input)
        st.session_state.input_key += 1  # Increment the key to reset the input field

    st.markdown("</div>", unsafe_allow_html=True)




        


