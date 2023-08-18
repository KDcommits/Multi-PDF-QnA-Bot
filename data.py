import os 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import FAISS


class Data:
    def __init__(self, pdf_data_path, vector_db_path):
        self.pdf_data_path = pdf_data_path
        self.vector_db_path = vector_db_path
        self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                            model_kwargs={'device':'cuda'})
    def createPDFVectorDB(self):
        loader = DirectoryLoader(self.pdf_data_path, glob='*.pdf', loader_cls=PyPDFLoader,
                                use_multithreading=True, show_progress=True)
        docs = loader.load()
        text_splitter  = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)
        db = FAISS.from_documents(texts, self.embedding_model)
        db.save_local(self.vector_db_path)

    def create_top_k_chunk(self, question, top_k):
        test_idex = FAISS.load_local(self.vector_db_path,self.embedding_model)
        top_k_chunks  = test_idex.similarity_search(question,k=top_k)
        return top_k_chunks
    

# pdf_data_path = ".\\Data\\Input\\Input File"
# pdf_vector_embedding_path = ".\\Vector_DB_PDF"
# data_obj = Data(pdf_data_path,pdf_vector_embedding_path)
# data_obj.createPDFVectorDB()
# question = 'Artificial Intelligence in budget?'
# top_k_chunks = data_obj.create_top_k_chunk(question, top_k =3)
# print(top_k_chunks)