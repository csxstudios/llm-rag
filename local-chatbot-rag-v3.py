from langchain_community.document_loaders import TextLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, request, session, jsonify
from dotenv import load_dotenv
import helpers

load_dotenv()

app = Flask(__name__)

# load the document and split it into chunks
loader = TextLoader("./data/NOTES.txt")
docs = loader.load()

retriever = helpers.DocsToChroma(docs, False)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm_local = helpers.GetLocalLLM("MODEL_MIXTRAL_7B")
llm_groq = helpers.GetGroqLLM("mixtral-8x7b-32768")

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
    llm_groq, retriever, contextualize_q_prompt
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
question_answer_chain = create_stuff_documents_chain(llm_groq, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

print("Chatbot initialized, ready to chat...")

# answer = conversational_rag_chain.invoke(
#     {"input": "What is a naked short put?"},
#     config={
#         "configurable": {"session_id": "abc123"}
#     },  # constructs a key "abc123" in `store`.
# )["answer"]

@app.route('/query', methods=['POST'])
def query():
    # global retrieval_chain
    print("Incoming query...")
    data = request.json
    query = data.get('query')
    print("Query: " + query)
    #local_answer=helpers.SearchChroma(query, retriever)
    res = conversational_rag_chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )
    sources = helpers.GetSources(res)
    return jsonify(answer=res['answer'], sources=sources), 200

if __name__ == '__main__':
    app.run(debug=True)