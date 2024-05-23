from langchain_community.document_loaders import TextLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.load.dump import dumps
from flask import Flask, request, session, jsonify, Response
from dotenv import load_dotenv
import helpers

load_dotenv()

app = Flask(__name__)

# load the document and split it into chunks
# docs = helpers.txt_to_doc("./data/NOTES.txt")

retriever = helpers.get_vectorstore().as_retriever(search_kwargs={'k': 3})

# llm_mixtral = helpers.GetLocalLLM("MODEL_MIXTRAL_7B")
# llm_llama3 = helpers.GetLocalLLM("MODEL_LLAMA3_8B")
llm_groq = helpers.get_groq_llm("mixtral-8x7b-32768")

llm = llm_groq

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
ONLY use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def clear_store():
    store.clear()
    print("Cleared chat history 'store'.")


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

print("Chatbot initialized, ready to chat...")

@app.route('/query', methods=['POST'])
def query():
    # global retrieval_chain
    print("Incoming query...")
    data = request.json
    ip = request.remote_addr
    print("data",data)
    query = data.get('query')
    session = data.get('session')
    helpers.append_to_log(query, session, ip)
    res = conversational_rag_chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": session}
        },
    )
    if "I don't" in res['answer'] or "large language model" in res['answer'] or "I cannot" in res['answer'] or "I am" in res['answer'] or "My " in res['answer']:
        clear_store()
        res = {
            "answer": "Unfortunately, I'm unable to provide an answer to your question at the moment. ",
            "feedback": "However, I'm here to assist you as best as I can. Your input is crucial to my learning and development! If you would like to submit feedback on information that may help answer this type of question in the future, please select the Submit Feedback button to help me out."
        }
        return jsonify(answer=res['answer']), 200
    else: 
        clear_store()
        print("Response: " + dumps(res))
        sources = helpers.get_sources(res)
        return jsonify(answer=res['answer'], sources=sources), 200
    
@app.route('/', methods=['GET'])
def index():  # pragma: no cover
    content = open('index.html').read()
    return Response(content, mimetype="text/html")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0',port=port)