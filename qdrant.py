from dotenv import load_dotenv
# import streamlit as st
from langchain.llms import HuggingFaceHub, OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
import qdrant_client
import os
import time  # Import the time module

# Define a function to generate summaries
def generate_summary(input_text, summarization_model):
    return summarization_model(input_text, max_length=200, temperature=0.5)

# ... (Rest of your code remains the same)
def get_vector_store_with_retry():
    max_retries = 3  # Maximum number of retries
    retry_delay = 1  # Delay between retries (in seconds)

    for retry in range(max_retries):
        try:
            client = qdrant_client.QdrantClient(
                os.getenv("QDRANT_HOST_URL"),
                api_key=os.getenv("QDRANT_API_KEY")
            )

            # You might need to adjust the embeddings and collection_name as per your requirements
            #os.getenv("HUGGINGFACE_API_KEY")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            

            vector_store = Qdrant(
                client=client,
                collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
                embeddings=embeddings,
            )

            vector_store.add_texts(texts)
            return vector_store

        except qdrant_client.http.exceptions.ResponseHandlingException as e:
            print(f"Attempt {retry + 1} failed with error: {e}")
            if retry < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Exiting.")
                raise  # If max retries reached, raise the exception
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# The rest of your code remains the same...
with open('C:\\Users\\krish\\OneDrive\\Desktop\\Study\\Gen AI\\Multi-PDF-QnA-Bot\\media\\Sony ZV-E1 Full-Frame Interchangeable-Lens Mirrorless vlog Camera.pdf', encoding="latin1") as f:
    raw_text = f.read()
print(raw_text)
texts = get_chunks(raw_text)

load_dotenv()
vector_store = get_vector_store_with_retry()
openAILLM = OpenAI(openai_api_key=os.getenv('OPENAI_KEY'),temperature=0,max_tokens=1024)
qa = RetrievalQA.from_chain_type(
        #llm=HuggingFaceHub(repo_id="google/tapas-base-finetuned-wtq", model_kwargs={"temperature":0.5, "max_length":512}),
        llm = openAILLM,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

question ='what are the cons of of Sony ZV E1 Camera?'
ans = qa.run(question)
print(ans)

# def main():
#     load_dotenv()
    
#     st.set_page_config(page_title="Ask Qdrant")
#     st.header("Ask your remote database 💬")
    
#     # create vector store with retry
#     vector_store = get_vector_store_with_retry()
    
#     # create chain 
#     os.getenv("HUGGINGFACEHUB_API_TOKEN")
#     openAILLM = OpenAI(openai_api_key=os.getenv('OPENAI_KEY'),temperature=0,max_tokens=1024)
#     #hfLLM  = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
#     qa = RetrievalQA.from_chain_type(
#         #llm=HuggingFaceHub(repo_id="google/tapas-base-finetuned-wtq", model_kwargs={"temperature":0.5, "max_length":512}),
#         llm = openAILLM,
#         chain_type="stuff",
#         retriever=vector_store.as_retriever()
#     )

#     # show user input
#     user_question = st.text_input("Ask a question about your PDF:")
#     if user_question:
#         st.write(f"Question: {user_question}")
#         answer = qa.run(user_question)
#         st.write(f"Answer: {answer}")
#         #st.write(f"Question: {user_question}")
#         # Add a summarization step
#         # os.getenv("HUGGINGFACEHUB_API_TOKEN")
#         # summarization = HuggingFaceHub(repo_id="facebook/bart-large-cnn", model_kwargs={"temperature":0.5, "max_length":200})
#         # summary = summarization(user_question, max_length=200, temperature=0.5)
        
#         # Split the user_question into chunks
#         #input_text_chunks = get_chunks(user_question)
        
#         # Initialize an empty list to store the summaries
#         # summaries = []

#         # Process each chunk and generate summaries
#         # for chunk in input_text_chunks:
#         #     # Generate a summary for each chunk
#         #     chunk_summary = generate_summary(chunk, summarization)
#         #     summaries.append(chunk_summary)

#         # Concatenate the summaries to get the final result
#         #summary = " ".join(summaries)

#         #st.write(f"Summary: {summary}")
        
#         #answer = qa.run(summary)
#         #st.write(f"Answer: {answer}")

# if __name__ == '__main__':
#     main()
