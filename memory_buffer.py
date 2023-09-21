import os
from langchain.chains import ConversationChain,LLMChain
from langchain.vectorstores import FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub,OpenAI
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
load_dotenv()

# ##loading documents
# loader = PyPDFLoader("E://langchain_memory//Reports//HP Laser 1008w Printer.pdf")
# pages = loader.load()

# ##creating chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100,chunk_overlap  = 20)
# texts = text_splitter.split_documents(pages)

# ##creating faiss vectorstore   
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# vectorstore = FAISS.from_documents(texts, embedding=embeddings)

##large language model
#llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.1, "max_length":512})
#llm = OpenAI(model_name='text-davinci-003',temperature=0,max_tokens=256)
llm=OpenAI(openai_api_key=os.getenv('OPENAI_KEY'))
##creating an instance of memory

#memory.save_context({"input": "hi"}, {"output": "whats up"})

template = """You are a nice chatbot having a conversation with a human.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""
prompt = PromptTemplate(input_variables=["chat_history", "question"], template=template)

memory = ConversationBufferMemory(memory_key="chat_history")
## getting conversation chain
# conversation = ConversationChain(
#     llm=llm, 
#     verbose=True, 
#     memory=memory,
#     prompt=prompt
# )
conversation = LLMChain(
    llm=llm, 
    verbose=True, 
    memory=memory,
    prompt=prompt
)
conv = conversation.run({"question": "Hi there my friend"})
print(conv)


