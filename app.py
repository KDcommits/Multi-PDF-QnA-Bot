from flask import Flask, request, jsonify, render_template
import openai,os
from model import Model
from data import Data
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
    file = request.files['file']
    if file:
        if not os.path.exists(pdf_path):
            os.makedirs(pdf_path)
        file.save(os.path.join(pdf_path,"temp_file.pdf"))
        ### Creating vector db out of the uploaded pdfs
        data_obj = Data(pdf_path, db_path)
        data_obj.createPDFVectorDB()

        return jsonify({"status": 201, 'message':'success'})
    else:
        error = 'Some Error Occured!'
        return jsonify({'error': error})


@app.route('/pdfchat', methods=['POST'])
def pdf_query():
    try:
        input_question = request.json['input_text']
        print(input_question)
        model_obj = Model()
        data_obj = Data(pdf_path, db_path)
        top_k_chunks = data_obj.create_top_k_chunk(input_question, top_k=3)
        prompt = model_obj.createQuestionPrompt(input_question, top_k_chunks)
        print(prompt)
        response = model_obj.generateAnswer(prompt)
        print(response) 
        return jsonify({'response': response})
    
    except Exception as e:
        print(str(e))
        return jsonify({"error":str(e)})


if __name__ == "__main__":
    app.run(debug=False)