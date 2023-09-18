import os 
import re
import time
import copy
import fitz
import numpy as np
import pinecone as pc
from tqdm.auto import tqdm
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS, Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader

load_dotenv()

class CustomTextSplitter:

    def __init__(self, chunk_size, chunk_overlap):
        self.keep_separator= False
        self.strip_whitespace=True
        self.is_separator_regex=False
        self.add_start_index=False
        self.length_function = len
        self.chunk_size=chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators=["\n\n", "\n", " ", ""]

    def _split_text_with_regex(self,text: str, separator: str):
        # Now that we have the separator, split the text
        if separator:
            if self.keep_separator:
                # The parentheses in the pattern keep the delimiters in the result.
                _splits = re.split(f"({separator})", text)
                splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = [_splits[0]] + splits
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]
    
    def _join_docs(self, docs, separator:str,):
        text = separator.join(docs)
        if self.strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text
        

    def _merge_splits(self, splits, separator: str):
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self.length_function(separator)

        docs = []
        current_doc= []
        total = 0
        for d in splits:
            _len = self.length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self.chunk_size
            ):
                if total > self.chunk_size:
                    print(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self.chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self.chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self.chunk_size
                        and total > 0
                    ):
                        total -= self.length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs
    
    def _split_text(self,text: str,):
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = self.separators[-1]
        new_separators = []
        for i, _s in enumerate(self.separators):
            _separator = _s if self.is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = self.separators[i + 1 :]
                break

        _separator = separator if self.is_separator_regex else re.escape(separator)
        splits = self._split_text_with_regex(text, _separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self.keep_separator else separator
        for s in splits:
            if self.length_function(s) < self.chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks
    

    def create_documents(self, texts, pdf_name):
        """Create documents from a list of texts."""
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self._split_text(text):
                if self.add_start_index:
                    index = text.find(chunk, index + 1)
                new_doc = Document(page_content=chunk, metadata={})
                new_doc.metadata['source'] = pdf_name
                documents.append(new_doc)
        return documents
    
class PdfTextExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path 
        self.start_page = 1
        self.end_page = None

    def _preprocess(self,text):
        '''
        preprocess extrcted text from pdf
        1. Replace new line character with whitespace.
        2. Replace redundant whitespace with a single whitespace
        '''
        text = text.replace('\n', ' ')
        text = re.sub('\s+', ' ', text)
        text = re.sub(r'\\u[e-f][0-9a-z]{3}',' ', text)
        return text
    
    def _pdf_to_text(self, pdf_filename):
        '''
            convert pdf to a list of words.
        '''
        doc = fitz.open(self.pdf_path)
        total_pages= doc.page_count

        if self.end_page is None:
            self.end_page = total_pages
        text_list=[]

        for i in tqdm(range(self.start_page-1, self.end_page)):
            text= doc.load_page(i).get_text('text')
            text= self._preprocess(text)
            text_list.append(text+ f' [{pdf_filename}, page:{i+1}]')
        doc.close()
        return text_list
    

class Data:

    def __init__(self, pdf_data_path, vector_db_path):

        self.pdf_data_path = pdf_data_path
        self.vector_db_path = vector_db_path
        self.pinecone_api_key = os.getenv('PINECONE_KEY')
        self.pinecone_env = os.getenv('PINECONE_ENV')
        self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.openai_embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002',
                                                        openai_api_key=os.getenv('OPENAI_KEY'))
        
    def createPDFVectorDBwithFAISS(self, chunk_size, chunk_overlap):
        document_list=[]
        for pdf_filename in os.listdir(self.pdf_data_path):
            pdf_file_path = os.path.join(self.pdf_data_path,pdf_filename)
            extracted_text_list = PdfTextExtractor(pdf_file_path)._pdf_to_text(pdf_filename)
            merged_text_list = ['.'.join(extracted_text_list)]
            splitter = CustomTextSplitter(chunk_size, chunk_overlap)
            docs  = splitter.create_documents(merged_text_list,pdf_filename)
            document_list.extend(docs)
    
        db = FAISS.from_documents(document_list, self.embedding_model)
        db.save_local(self.vector_db_path)

    def create_top_k_chunk_from_FAISS(self, question,top_k):
        test_idex = FAISS.load_local(self.vector_db_path,self.embedding_model)
        top_k_chunks  = test_idex.similarity_search(question,k=top_k)
        return top_k_chunks
    
    def fetch_FAISS_VectorDB(self):
        test_index = FAISS.load_local(self.vector_db_path,self.embedding_model)
        return test_index


    def createPDFVectorDBwithPinecone(self,chunk_size, chunk_overlap):
        document_list=[]
        for pdf_filename in os.listdir(self.pdf_data_path):
            pdf_file_path = os.path.join(self.pdf_data_path,pdf_filename)
            extracted_text_list = PdfTextExtractor(pdf_file_path)._pdf_to_text(pdf_filename)
            merged_text_list = ['.'.join(extracted_text_list)]
            splitter = CustomTextSplitter(chunk_size, chunk_overlap)
            docs  = splitter.create_documents(merged_text_list,pdf_filename)
            document_list.extend(docs)

        embeddings = []
        ids = []
        metadatas = []
        for i in range(len(document_list)):
            if i%5==0:
                time.sleep(5)
            page_content = document_list[i].page_content
            source_pdf = document_list[i].metadata['source'].split('\\')[-1]
            embedded_page_content = self.openai_embedding_model.embed_query(page_content)
            metadata = {
                'source' : source_pdf,
                'page_content' : page_content
            }
            ids.append(str(i))
            embeddings.append(embedded_page_content)
            metadatas.append(metadata)

        pc.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
        index = pc.Index('pdf-index')
        index.upsert(vectors = zip(ids, embeddings, metadatas))

    def create_top_k_chunk_from_Pinecone(self, question,top_k):
        pc.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
        index = pc.Index('pdf-index')
        vectorstore = Pinecone( index, self.openai_embedding_model.embed_query, text_key='page_content')
        top_k_chunks = vectorstore.similarity_search(question, k=top_k)
        return top_k_chunks


# pdf_data_path = ".\\media"
# pdf_vector_embedding_path = ".\\VectorDB"
# data_obj = Data(pdf_data_path,pdf_vector_embedding_path)
# data_obj.createPDFVectorDBwithFAISS(chunk_size=2000, chunk_overlap=500)
# # data_obj.createPDFVectorDBwithPinecone(chunk_size=2000, chunk_overlap=500)
# test_question = "Find the cost of Sony ZV-E1 Full Frame camera"
# result = data_obj.create_top_k_chunk_from_FAISS(test_question, top_k =3)
# # result = data_obj.create_top_k_chunk_from_Pinecone(test_question, top_k =3)
# print(result)
# print(result[0].metadata['source'])
# print(result[0].metadata['page'])
# print(result[0].page_content)
