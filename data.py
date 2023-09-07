import os 
import time
import pinecone as pc
from dotenv import load_dotenv
from langchain.vectorstores import FAISS, Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

load_dotenv()

class Data:
    def __init__(self, pdf_data_path, vector_db_path):
        self.pdf_data_path = pdf_data_path
        self.vector_db_path = vector_db_path
        self.pinecone_api_key = os.getenv('PINECONE_KEY')
        self.pinecone_env = os.getenv('PINECONE_ENV')
        self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.openai_embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002',
                                                       openai_api_key=os.getenv('OPENAI_KEY'))

    def createPDFVectorDBwithFAISS(self):
        loader = DirectoryLoader(self.pdf_data_path, glob='*.pdf', loader_cls=PyPDFLoader,
                                use_multithreading=True, show_progress=True)
        docs = loader.load()
        text_splitter  = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50, 
                                                        separators=["\n\n", "\n", " ", ""])
        texts = text_splitter.split_documents(docs)
        db = FAISS.from_documents(texts, self.embedding_model)
        db.save_local(self.vector_db_path)

    def create_top_k_chunk_from_FAISS(self, question, top_k):
        test_idex = FAISS.load_local(self.vector_db_path,self.embedding_model)
        top_k_chunks  = test_idex.similarity_search(question,k=top_k)
        return top_k_chunks
    

    def createPDFVectorDBwithPinecone(self):
        loader = DirectoryLoader(self.pdf_data_path, glob='*.pdf', loader_cls=PyPDFLoader,
                                use_multithreading=True, show_progress=True)
        docs = loader.load()
        text_splitter  = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200, 
                                                        separators=["\n\n", "\n", " ", ""])
        texts = text_splitter.split_documents(docs)
        embeddings = []
        ids = []
        metadatas = []
        for i in range(len(texts)):
            if i%5==0:
                time.sleep(5)
            page_content = texts[i].page_content
            source_pdf = texts[i].metadata['source'].split('\\')[-1]
            page_number = str(texts[i].metadata['page'])
            embedded_page_content = self.openai_embedding_model.embed_query(page_content)
            metadata = {
                'source' : source_pdf,
                'page' : page_number,
                'page_content' : page_content
            }
            ids.append(str(i))
            embeddings.append(embedded_page_content)
            metadatas.append(metadata)

        pc.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
        index = pc.Index('pdf-index')
        index.upsert(vectors = zip(ids, embeddings, metadatas))


    def create_top_k_chunk_from_Pinecone(self, question, top_k):
        pc.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
        index = pc.Index('pdf-index')
        #embedded_question = self.embedding_model.embed_query(question)
        vectorstore = Pinecone( index, self.openai_embedding_model.embed_query, text_key='page_content')
        top_k_chunks = vectorstore.similarity_search(question, k=top_k)
        #result = index.query(embedded_question, top_k = top_k, includeMetaData=True)
        return top_k_chunks


    

# pdf_data_path = ".\\media"
# pdf_vector_embedding_path = ".\\VectorDB"
# data_obj = Data(pdf_data_path,pdf_vector_embedding_path)
# data_obj.createPDFVectorDBwithPinecone()
# test_question = "Find the cost of Sony ZV-E1 Full Frame camera"
# result = data_obj.create_top_k_chunk_from_Pinecone(test_question, top_k =3)
# print(result[0].metadata['source'])
# print(result[0].metadata['page'])
# print(result[0].page_content)


# index = pc.Index('pdf-index')
# encoded_sentence = model.encode(test_question)
# result = index.query(encoded_sentence, top_k =3, includeMetaData=True)

# loader = DirectoryLoader(pdf_data_path, glob='*.pdf', loader_cls=PyPDFLoader,
#                                 use_multithreading=True, show_progress=True)
# docs = loader.load()
# text_splitter  = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
# texts = text_splitter.split_documents(docs)
# print(texts[0].metadata['source'].split('\\')[-1])
# print(str(texts[0].metadata['page']))
# print(str(texts[0].page_content))
# print('['+top_k_chunks[i].metadata['source'].split('\\')[-1] + " , page : " + str(top_k_chunks[i].metadata['page'])+'] ')
# data_obj = Data(pdf_data_path,pdf_vector_embedding_path)
# data_obj.createPDFVectorDB()
# question = 'Artificial Intelligence in budget?'
# top_k_chunks = data_obj.create_top_k_chunk(question, top_k =3)
# print(top_k_chunks)