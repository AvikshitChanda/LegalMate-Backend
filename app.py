import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import html
from dataclasses import dataclass
from typing import Literal
import os
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import time

HUGGINGFACEHUB_API_TOKEN = st.secrets['HUGGINGFACEHUB_API_TOKEN']

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["👤 Human", "👨🏻‍⚖️ Ai"]
    message: str


def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state:
        llama = LlamaAPI(st.secrets["LlamaAPI"])
        model = ChatLlamaAPI(client=llama)
        chat = ChatGroq(temperature=0.5, groq_api_key=st.secrets["Groq_api"], model_name="mixtral-8x7b-32768")

        embeddings = download_hugging_face_embeddings()

        # Initializing the Pinecone
        pinecone.init(
            api_key=st.secrets["PINECONE_API_KEY"],  # find at app.pinecone.io
            environment=st.secrets["PINECONE_API_ENV"]  # next to api key in console
        )
        index_name = "legal-advisor"  # put in the name of your pinecone index here

        docsearch = Pinecone.from_existing_index(index_name, embeddings)

        prompt_template = """
            You are a trained bot to guide people about Indian Law. You will answer user's query with your knowledge and the context provided. 
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            Also check if the given context answer the question asked if not don't answer using that context.
            Do not say thank you and tell you are an AI Assistant and be open about everything.
            Use the following pieces of context to answer the users question.
            Give very detailed answer. 
            Context: {context}
            Question: {question}
            Only return the helpful answer below and nothing else.
            Helpful answer:
            """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
            )
        retrieval_chain = ConversationalRetrievalChain.from_llm(llm=chat,
                                                      chain_type="stuff",
                                                      retriever=docsearch.as_retriever(
                                                          search_kwargs={'k': 2}),
                                                      return_source_documents=True,
                                                      combine_docs_chain_kwargs={"prompt": PROMPT},
                                                      memory= memory
                                                     )

        st.session_state.conversation = retrieval_chain


def on_click_callback():
    human_prompt = st.session_state.human_prompt
    st.session_state.human_prompt = ""
    response = st.session_state.conversation(human_prompt)
    llm_response = response['answer']
    st.session_state.history.append(Message("👤 Human", human_prompt))
    st.session_state.history.append(Message("👨🏻‍⚖️ Ai", llm_response))


initialize_session_state()

st.title("LegalEase Advisor Chatbot 🇮🇳")

st.markdown(
    """
    👋 **Namaste! Welcome to LegalEase Advisor!**
    I'm here to assist you with your legal queries within the framework of Indian law. Whether you're navigating through specific legal issues or seeking general advice, I'm here to help.
    
    📚 **How I Can Assist:**
    
    - Answer questions on various aspects of Indian law.
    - Guide you through legal processes relevant to India.
    - Provide information on your rights and responsibilities as per Indian legal standards.
    
    ⚖️ **Disclaimer:**
    
    While I can provide general information, it's essential to consult with a qualified Indian attorney for advice tailored to your specific situation.
    
    🤖 **Getting Started:**
    
    Feel free to ask any legal question related to Indian law, using keywords like "property rights," "labor laws," or "family law." I'm here to assist you!
    Let's get started! How can I assist you today?
    """
)

if "history" not in st.session_state:
    st.session_state.history = []

if "generated" not in st.session_state:
    st.session_state.generated = []

if "past" not in st.session_state:
    st.session_state.past = []

chat_placeholder = st.empty()

with chat_placeholder.container():
    for i, chat in enumerate(st.session_state.history):
        if chat.origin == "👤 Human":
            message(chat.message, is_user=True, key=f"{i}_user")
        else:
            message(chat.message, key=f"{i}")

def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    st.session_state.generated.append({"type": "normal", "data": f"The message from Bot\nWith new line\n{user_input}"})
    st.session_state.history.append(Message("👤 Human", user_input))
    st.session_state.history.append(Message("👨🏻‍⚖️ Ai", f"The message from Bot\nWith new line\n{user_input}"))

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]
    del st.session_state.history[:]

with st.container():
    st.text_input("User Input:", on_change=on_input_change, key="user_input")
    st.button("Clear message", on_click=on_btn_click)
