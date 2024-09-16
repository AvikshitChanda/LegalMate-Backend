import os
import pdfplumber
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
def extract_text_from_pdfs(directory):
    documents = []
    # Loop over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            
            # Open the PDF
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text()
            
             # Append extracted text with metadata as a Document
            documents.append({
                "content": full_text,
                "metadata": {"source": filename}
            })
    
    return documents

# # Specify the directory where your PDFs are stored
pdf_directory = "dataset"  # Replace with the path to your dataset folder
pdf_documents = extract_text_from_pdfs(pdf_directory)



def create_langchain_documents(extracted_documents):
    langchain_documents = []
    
    for doc in extracted_documents:
        langchain_documents.append(
            Document(page_content=doc["content"], metadata=doc["metadata"])
        )
    
    return langchain_documents

# Convert the extracted PDFs to LangChain documents
documents = create_langchain_documents(pdf_documents)
print(documents)


# # Step 1: Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# # Step 2: Initialize Chroma with persistence
docsearch = Chroma(
    collection_name="legal-advisor",
    embedding_function=embeddings,
    persist_directory="./chroma_db"  # Directory to store ChromaDB
)

# Step 3: Add the documents to ChromaDB
docsearch.add_documents(documents)

# Step 4: Persist the ChromaDB to disk
docsearch.persist()

query = "What are the key aspects of the Indian Police Conduct Code?"
results = docsearch.similarity_search(query, k=2)

for result in results:
    print(result.page_content)
    print(result.metadata)
