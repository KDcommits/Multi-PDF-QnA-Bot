import os 
import re
import openai
from dotenv import load_dotenv

load_dotenv()

class GenerativeModel:

    def _createQuestionPrompt(self, question: str, top_k_chunks:list) ->str:
        '''
            Create prompt to generate response as per requirement, precise and contextual.
            Also we are trying stopping hallucinations by making the prompt strict.
        '''
        prompt= ""
        prompt += 'search results:\n\n'
        for i in range(len(top_k_chunks)):
            meta_info = '['+top_k_chunks[i].metadata['source'].split('/')[-1] + " , page : " + str(top_k_chunks[i].metadata['page'])+'] '
            page_content = top_k_chunks[i].page_content.replace('\n', ' ')
            page_content = re.sub('\s+', ' ', page_content)
            combined_content = meta_info + page_content
            prompt = combined_content +'\n\n'
        prompt += "Instructions: Compose a comprehensive reply to the query using the search results given."\
                    "Cite each reference using [pdfname.pdf , page : number] notation (every result has this number at the beginning)."\
                    "Citation should be done at the end of each sentence. If the search results mention multiple subjects"\
                    "with the same name, create separate answers for each. Only include information found in the results and"\
                    "don't add any additional information. Make sure the answer is correct and don't output false content."\
                    "If the text does not relate to the query, simply state 'Found Nothing'. Don't write 'Answer:'"\
                    "Directly start the answer.\n"
        prompt+= f"Query : {question} \n\n"
        return prompt
    
    def generate_response(self,prompt : str, engine ='text-davinci-003')->str:
        '''
            Harnessing the generative capability of the openAI model to generate the answer to the user's
            question in much human-comprehensive form. 
        '''

        openai.api_key = os.getenv('OPENAI_API_KEY')
        completions = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=2048,
            temperature=0.01,
        )
        answer = completions.choices[0]['text']
        return answer
    

