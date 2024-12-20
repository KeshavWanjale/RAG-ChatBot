import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Function to generate answers using Generative AI
def generate_answer(prompt):
    try:
        genai.configure(api_key=os.getenv("GEMEINI_API_KEY"))
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        answer = model.generate_content(prompt)
        return answer.text
    except Exception as e:
        return f"Error generating answer: {e}"

# Function to retrieve relevant context from the database
def get_relevant_context_from_db(query):
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
        search_results = vector_db.similarity_search(query, k=6)
        context = "\n".join(result.page_content for result in search_results)
        return context
    except Exception as e:
        return f"Error fetching context: {e}"

# Function to generate the RAG prompt
def generate_rag_prompt(query, context):
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""
    You are a helpful and informative bot that answers questions using text from the reference context included below. 
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
    strike a friendly and conversational tone. 
    If the context is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    CONTEXT: '{escaped}'
    
    ANSWER:
    """
    return prompt

# Streamlit App
st.title("Interactive Chatbot with RAG and Generative AI")
st.write("Enter your query below, and the chatbot will generate an answer using relevant context.")

# Input query
query = st.text_input("Your Query:", "")

if st.button("Submit"):
    if query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        # Fetch context and generate prompt
        with st.spinner("Fetching relevant context..."):
            context = get_relevant_context_from_db(query)
        with st.spinner("Generating response..."):
            prompt = generate_rag_prompt(query, context)
            answer = generate_answer(prompt)
        # Display results
        st.subheader("Context Retrieved:")
        st.text_area("Context", context, height=150)
        st.subheader("Chatbot Response:")
        st.write(answer)
