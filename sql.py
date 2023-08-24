import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

class SQLQuery:
    def __init__(self):
        self.DB_USERNAME = os.getenv('DB_USERNAME')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD')
        self.DB_HOST = os.getenv('DB_HOST')
        self.DB_NAME = os.getenv('DB_NAME')

    def createDBConnectionString(self):
        db_user = self.DB_USERNAME
        db_Password  = self.DB_PASSWORD
        db_host = self.DB_HOST
        db_name = self.DB_NAME
        connectionString = f"mysql+pymysql://{db_user}:{db_Password}@{db_host}/{db_name}"
        return connectionString

    def createAgentExecutor(self, openAI_model_name="gpt-3.5-turbo"):
        
        mysql_connection_string = self.createDBConnectionString()
        llm = ChatOpenAI(model_name= openAI_model_name )
        db = SQLDatabase.from_uri(mysql_connection_string)
        toolkit = SQLDatabaseToolkit(db=db, llm =llm)
        agent_executor = create_sql_agent(
                                        llm=llm,
                                        toolkit=toolkit,
                                        verbose=True,
                                        return_intermediate_steps=False)
        return agent_executor

    def fetchQueryResult(self,question):
        db_agent = self.createAgentExecutor()
        prompt = f'''You are a professional SQL Data Analyst whose job is to fetch results from the SQL database.\
            The question will be asked in # delimiter. If you are not able to find the answer write "Found Nothing" in response.\
            Do not write anything out of context or on your own.\
            Question : # {question} #'''
        db_agent.return_intermediate_steps=True
        agent_response = db_agent(prompt)
        output = agent_response['output']
        # query = agent_response['intermediate_steps'][-1][0].log.split('\n')[-1].split('Action Input:')[-1].strip().strip('"')
        return output 


# question = "Artificial intelligence in budget ?"
# question = "Find the email id of the salesRepEmployeeNumber for customer number 124"
# sql_obj = SQLQuery()
# response = sql_obj.fetchQueryResult(question )
# print(response)
