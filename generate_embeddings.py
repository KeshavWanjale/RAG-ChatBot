from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma


# Loading the PDF Document
loaders = [PyPDFLoader('./Python-Tutorial.pdf')]
docs = []
for file in loaders:
    docs.extend(file.load())

# Splitting the Text into Smaller Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# Generating Embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={'device': 'cpu'}
)

# Creating a Vector Store
vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_nccn")

print(vectorstore._collection.count())
