from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

class PdfData:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2', 
                                                model_kwargs={'device':'cuda'})  
        
    def storePDFVectorEmbeddings(self, pdf_data_path:str, pdf_vector_embedding_path:str)-> None:
        '''
            Reads multiple pdf's from the data_path and stores embeddings of the pdf content 
            in the vector_embedding_path. 
        '''
        loader = DirectoryLoader(pdf_data_path, glob='*.pdf', loader_cls=PyPDFLoader,
                                use_multithreading=True, show_progress=True)
        docs = loader.load()
        text_splitter  = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)
        db = FAISS.from_documents(texts, self.embeddings)
        db.save_local(pdf_vector_embedding_path)

    def fetch_top_chunks(self, pdf_vector_embedding_path:str,
                         question : str, top_k=3) ->list:
        '''
            Returns a list of langchain Document object.
            top_k is default set to 3 but it will be governed by user input(or may be not)
        '''
        test_idex = FAISS.load_local(pdf_vector_embedding_path,self.embeddings)
        top_k_chunks  = test_idex.similarity_search(question,k=top_k)
        return top_k_chunks


