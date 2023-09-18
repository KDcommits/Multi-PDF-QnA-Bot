import os 
import re
import openai
from dotenv import load_dotenv
from langchain.chains import LLMChain, ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory,ConversationSummaryMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

load_dotenv()

class Model:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_KEY")
        self.model = 'text-davinci-003'
        self.llm = ChatOpenAI( openai_api_key = os.getenv("OPENAI_KEY"))
        self.chat_history = []

    def createQuestionPrompt(self, top_k_chunks):
        prompt= ""
        prompt += 'Product Details:\n\n'
        for i in range(len(top_k_chunks)):
            #meta_info = '['+top_k_chunks[i].metadata['source'].split('\\')[-1] + " , page : " + str(top_k_chunks[i].metadata['page'])+'] '
            page_content = top_k_chunks[i].page_content.replace('\n', ' ')
            page_content = re.sub('\s+', ' ', page_content)
            # combined_content = meta_info + page_content
            prompt += page_content +'\n\n'
        # if len(chat_history)>0:
        #     historical_interaction = ".".join(chat_history)
        # else:
        #     historical_interaction = ""
        # prompt += 'Chat History: '+ historical_interaction
        prompt += '''\nInstructions: Compose a comprehensive reply to the query using the product details and chat history given.
        Cite each reference using [pdfname.pdf , page : number] notation (every result has this number at the beginning).
        Citation should be done at the end of each sentence. If the search results mention multiple subjects
        with the same name, create separate answers for each. Only include information found in the results and
        don't add any additional information. Make sure the answer is correct and don't output false content.
        If the text does not relate to the query, simply state 'Found Nothing'. Don't write 'Answer:'
        Directly start the answer.\n'''.replace('\\n',' ')
        #prompt+= f"Query : {question} \n\n"

        return prompt
    
    def createQuestionPromptTemplate(self, prompt):
        prompt_template_llmchain = ChatPromptTemplate.from_messages([
                    SystemMessage(content=f"You are a QnA Chatbot whose job is to greet the user politely and asnwer to the question they ask."+prompt), 
                    MessagesPlaceholder(variable_name="chat_history"),         # Where the memory will be stored.
                    HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injected
                 ])
        
        # prompt_template_conversationchain = PromptTemplate(input_variables=["product_details", "human_input"], template=prompt+'''Relevant entity information:
        #                                                                                                                                                     {product_details}
        #                                                                                                                                                     Conversation:
        #                                                                                                                                                     Human: {human_input}
        #                                                                                                                                                     AI:''')
        # prompt_template_conversationchain = ChatPromptTemplate.from_messages([
        #             SystemMessage(content=f"You are a QnA Chatbot whose job is to greet the user politely and asnwer to the question they ask."+prompt+'''Relevant entity information:
        #                                                                                                                                                     {product_details}
        #                                                                                                                                                     Conversation:
        #                                                                                                                                                     Human: {human_input}
        #                                                                                                                                                     AI:'''), 
        #             MessagesPlaceholder(variable_name="product_details"),         # Where the memory will be stored.
        #             HumanMessagePromptTemplate.from_template("{human_input}"),    # Where the human input will injected
        #          ])
        return prompt_template_llmchain

    def generateAnswer(self, prompt):
        openai.api_key = self.openai_api_key
        completions = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=1024,
            temperature=0,
        )
        answer = completions.choices[0]['text']
        return answer
    
    def generateAnswerwithMemory(self,question, prompt_template, chat_history):
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=2)
        llm = ChatOpenAI( openai_api_key = os.getenv("OPENAI_KEY"))
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            verbose=True,
            memory=memory,
        )

        llm_chain_response = llm_chain.predict(human_input = question)

        # conversation_chain = ConversationChain(
        #     llm=llm,
        #     prompt=prompt_template,
        #     verbose=True,
        #     memory=memory
        # )

        # conversation_chain_response = conversation_chain.predict(human_input= question)
        interaction = 'human:'+question+'\nchatbot:'+llm_chain_response
        chat_history.append(interaction)
        return llm_chain_response
    
    def generateAnswerwithConversationSummary(self, question, vectorstore,chat_history):
        
        memory = ConversationSummaryMemory(llm=self.llm, memory_key='chat_history',
                                            return_messages=True,output_key='answer')

        qa = ConversationalRetrievalChain.from_llm(self.llm, vectorstore.as_retriever(), 
                                                    return_source_documents=True, 
                                                    memory=memory, get_chat_history=lambda h :h,)
        
        prompt = """Instructions: Compose a comprehensive reply to the query using the product details given.
        Cite each reference using [pdfname.pdf , page : number] notation (every result has this number at the beginning).
        Citation should be done at the end of each sentence. If the search results mention multiple subjects
        with the same name, create separate answers for each. Only include information found in the results and
        don't add any additional information. Make sure the answer is correct and don't output false content.
        If the text does not relate to the query, simply state 'Found Nothing'. Don't write 'Answer:'
        Directly start the answer.\n""".replace('\n','')
        prompt+= f"Query : {question} \n\n"

        chat_history_tuples = []
        if len(chat_history)>0:
            for message in self.chat_history:
                chat_history_tuples.append((message[0], message[1]))
        # Get result from QA chain
        result = qa({"question": prompt, "chat_history": chat_history_tuples})
        # Append user message and response to chat history
        chat_history.append((question, result["answer"]))
        print(chat_history)
        return chat_history[-1][1]





