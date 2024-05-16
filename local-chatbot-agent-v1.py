from langchain_community.document_loaders import TextLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.load.dump import dumps
from flask import Flask, request, session, jsonify
from dotenv import load_dotenv
import helpers
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.agent_toolkits import create_sql_agent
import re

load_dotenv()

app = Flask(__name__)

db = SQLDatabase.from_uri("sqlite:///sql/oscars.db")
# print(db.dialect)
# print(db.get_usable_table_names())
# results=db.run("SELECT COUNT(*) FROM oscars;")
# print(results)

# llm_mixtral = helpers.get_local_llm("MODEL_MIXTRAL_7B")
# llm_llama3 = helpers.get_local_llm("MODEL_LLAMA3_8B")
llm_groq = helpers.get_groq_llm("mixtral-8x7b-32768")
# llm_groq = helpers.get_groq_llm("llama3-8b-8192")
# llm_groq = helpers.get_groq_llm("gemma-7b-it")

llm = llm_groq

questions = [
    "What was Lady Gaga nominated for?",
    "Which category has the most nominations"
    ]

chain = create_sql_query_chain(llm, db)

for question in questions:
    response = chain.invoke({"question": question})
    print("response", response)
    sql = re.search("(SELECT[\s\S][^`]+)", response)
    print("sql", sql)
    if len(sql[0]) > 0:
        print("run sql", sql[0])
        results = db.run(sql[0])
        print("SQL Result: ", results)
    else:
        print("No sql to run")
    wait = input("Press Enter to continue.")

# execute_query = QuerySQLDataBaseTool(db=db)
# write_query = create_sql_query_chain(llm, db)
# #print("sql query", write_query)
# chain = write_query | execute_query

# answer_prompt = PromptTemplate.from_template(
#     """Given the following user question, corresponding SQL query, and SQL result, answer the user question using the "oscars" table.

# Question: {question}
# SQL Query: {query}
# SQL Result: {result}
# Answer: """
# )

# answer = answer_prompt | llm | StrOutputParser()
# chain = (
#     RunnablePassthrough.assign(query=write_query).assign(
#         result=itemgetter("query") | execute_query
#     )
#     | answer
# )
# response=chain.invoke({"question": "Which category had the most winners?"})
# print("response", response)

# agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True,max_execution_time=1)
# agent_executor.invoke(
#     {
#         "input": "Which category had the most winners?"
#     }
# )



# print("Chatbot initialized, ready to chat...")

# @app.route('/query', methods=['POST'])
# def query():
#     # global retrieval_chain
#     print("Incoming query...")
#     data = request.json
#     query = data.get('query')
#     print("Query: " + query)
#     res = db.run(response)
#     print("Response: " + dumps(res))
#     # sources = helpers.get_sources(res)
#     return jsonify(answer=res['answer'], sources=sources), 200

# if __name__ == '__main__':
#     app.run(debug=True)