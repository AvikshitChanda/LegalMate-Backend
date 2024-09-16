from dataclasses import dataclass
from typing import Literal
from flask_cors import CORS

from flask import Flask, request, jsonify
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq

app = Flask(__name__)
CORS(app)
@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["👤 Human", "👨🏻‍⚖️ Ai"]
    message: str

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

# Initialize the session state
def initialize_session_state():
    if "history" not in app.config:
        app.config["history"] = []
    if "conversation" not in app.config:
        llama = LlamaAPI("d9f5e52518c6432f9c568a308d17554b767f2543a040477d9888bf9704d4d065")
        model = ChatLlamaAPI(client=llama)
        chat = ChatGroq(temperature=0.5, groq_api_key="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", model_name="mixtral-8x7b-32768")

        embeddings = download_hugging_face_embeddings()

        # Initialize ChromaDB from persisted directory
        persist_directory = "chroma-db"  # This is the folder where Chroma will store and load data
        chroma_client = Chroma(
            collection_name="legal-advisor",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

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

        # Initialize the retrieval chain with memory
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        # Create the retrieval chain
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=chat,
            chain_type="stuff",
            retriever=chroma_client.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            memory=memory
        )

        app.config["conversation"] = retrieval_chain

initialize_session_state()

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    human_prompt = data.get('question', '')
    
    if not human_prompt:
        return jsonify({"error": "No question provided"}), 400

    # Get response from the conversation chain
    response = app.config["conversation"](human_prompt)
    llm_response = response['answer']
    
    # Update history
    app.config["history"].append(
        Message("👤 Human", human_prompt)
    )
    app.config["history"].append(
        Message("👨🏻‍⚖️ Ai", llm_response)
    )

    return jsonify({"answer": llm_response})

if __name__ == "__main__":
    app.run(debug=True)