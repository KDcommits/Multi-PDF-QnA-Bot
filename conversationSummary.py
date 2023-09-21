from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
import gradio as gr
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS


import os
os.environ["OPENAI_API_KEY"] = 


llm = ChatOpenAI(temperature=0)


#Load data in the reports folder
pdf_folder_path = f'./C:/Users/Lenovo/Downloads/Anugya Shaw/GenAI_Projects/pdfapp_gradio/Reports'
#pdf_loader = DirectoryLoader('./C:/Users/Lenovo/Downloads/Anugya Shaw/GenAI_Projects/pdfapp_gradio/Reports', glob="**/*.pdf")
#txt_loader = DirectoryLoader('./Reports/', glob="**/*.txt")
#word_loader = DirectoryLoader('./Reports/', glob="**/*.docx")

#loaders = [pdf_loader] #txt_loader, word_loader
#documents = []
#for loader in loaders:
    #documents.extend(loader.load())

loader = PyPDFDirectoryLoader(pdf_folder_path)
documents = loader.load()    


#chunk the data, create embeddings and store it into a vectorstore
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

#memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", max_token_limit=100)
memory = ConversationSummaryMemory(llm=llm, memory_key='chat_history',
                                    return_messages=True,output_key='answer')


#calling langchain 
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True, 
                                           memory=memory, get_chat_history=lambda h :h,)

#gradio UI for chatbot


# Define chat interface
css="footer {visibility: hidden}"
with gr.Blocks(css=css) as demo:
    gr.Markdown("""<h1><center> Chat with PDFs </center></h1>""")
    chatbot = gr.Chatbot()
    #msg = gr.Textbox()
    msg = gr.Textbox(show_label=False, placeholder="Ask a question and press enter.").style(container=False)
    submit = gr.Button("Submit")
    clear = gr.Button("Clear")
    #state = gr.State()
    chat_history = []
    

    def user(query, chat_history):
        print("User query:", query)
        print("Chat history:", chat_history)

        # Convert chat history to list of tuples
        chat_history_tuples = []
        for message in chat_history:
            chat_history_tuples.append((message[0], message[1]))

        # Get result from QA chain
        result = qa({"question": query, "chat_history": chat_history_tuples})

        # Append user message and response to chat history
        chat_history.append((query, result["answer"]))
        print("Updated chat history:", chat_history)

        return gr.update(value=""), chat_history


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True)
