from flask import Flask, request, jsonify, render_template
import openai,os
import shutil
from model2 import Model
from data2 import Data
from sql import SQLQuery
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")
pdf_path = os.path.join(os.getcwd(),'media')
db_path = os.path.join(os.getcwd(),'vectorDB')

app = Flask(__name__)

@app.route('/')
def pdfchat():
    return render_template('pdfchat.html')


@app.route('/pdfupload', methods=['POST'])
def upload_pdf():
    files = request.files.getlist('files')
    print(files)
    shutil.rmtree(pdf_path)
    os.makedirs(pdf_path)
    for file in files:
        if file:
            print(file.filename)
            if not os.path.exists(pdf_path):
                os.makedirs(pdf_path)
            file.save(os.path.join(pdf_path,file.filename))
        else:
            error = 'Some Error Occured!'
            return jsonify({'error': error})
    ### Creating vector db out of the uploaded pdfs
    try:
        data_obj = Data(pdf_path, db_path)
        shutil.rmtree(db_path)
        os.makedirs(db_path)
        data_obj.createPDFVectorDBwithFAISS(chunk_size=2000, chunk_overlap=500)
        return jsonify({"status": 201, 'message':'success'})
    except Exception as e:
        return jsonify({'error': e})

chat_history = []
@app.route('/pdfchat', methods=['POST'])
def pdf_query():
    global chat_history 
    try:
        input_question = request.json['input_text']
        print(input_question)
        model_obj = Model()
        # sql_obj = SQLQuery()
        data_obj = Data(pdf_path, db_path)
        #top_k_chunks = data_obj.create_top_k_chunk_from_Pinecone(input_question, top_k=2)
        #prompt = model_obj.createQuestionPrompt(top_k_chunks)
        #prompt_template = model_obj.createQuestionPromptTemplate(prompt=prompt)
        vectorstore = data_obj.fetch_FAISS_VectorDB()
        #print(prompt)
        response_pdf = model_obj.generateAnswerwithConversationSummary(input_question, vectorstore,chat_history)
        # response_sql = sql_obj.fetchQueryResult(input_question)
        print(response_pdf)
        # print(response_sql)
        # pdf_valid_response = response_pdf.__contains__("Found Nothing") ## Returns True/False
        # sql_valid_response = response_sql.__contains__("Found Nothing") ## Returns True/False
        # if (pdf_valid_response==True) & (sql_valid_response==False):
        #     return jsonify({'response': response_sql})
        # elif (pdf_valid_response==False) & (sql_valid_response==True):
        #     return jsonify({'response': response_pdf})
        # elif (sql_valid_response==False) & (pdf_valid_response==False):
        #     return jsonify({'response': "Response from DataFrame : "+ response_sql+"\n\n+"
        #                     "Response from pdf knowledgebase : "+ response_pdf})
        # else:
        #     return jsonify({'response': 'Sorry!!! The bot could not find the answer. Please be more precise on the query.'})

        return jsonify({'response': response_pdf})
    
    except Exception as e:
        print(str(e))
        return jsonify({"error":str(e)})


if __name__ == "__main__":
    app.run(debug=False)