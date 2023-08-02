import os 
from data import PdfData
from dotenv import load_dotenv
from model import GenerativeModel
from flask import Flask, render_template, request

load_dotenv()

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def index():
    response=None
    if request.method == 'POST':
        pdf_data_path = os.getenv('pdf_data_path')
        pdf_vector_embedding_path = os.getenv('pdf_vector_embedding_path')
        question = request.form['question']
        print(question)
        data_obj = PdfData()
        # data_obj.storePDFVectorEmbeddings(pdf_data_path, pdf_vector_embedding_path)
        model_obj = GenerativeModel()
        top_k_chunks = data_obj.fetch_top_chunks(pdf_vector_embedding_path, question, top_k=3)
        print(top_k_chunks)
        prompt = model_obj._createQuestionPrompt(question, top_k_chunks)
        #response = model_obj.generate_response(prompt)
        response ="hello"
        
    return render_template('index.html', answer = response)
    

if __name__ == '__main__':
    app.run(debug=True)

