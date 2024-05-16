from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, request, session, jsonify
from dotenv import load_dotenv
import helpers

load_dotenv()

app = Flask(__name__)

# load the document and split it into chunks
docs = helpers.txt_to_doc("./data/NOTES.txt")

retriever = helpers.docs_to_chroma(docs, False).as_retriever(search_kwargs={'k': 3})

# Initialize message history for conversation
message_history = ChatMessageHistory()

# Memory for conversational context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    chat_memory=message_history,
    return_messages=True,
)

llm_local = helpers.get_local_llm("MODEL_MIXTRAL_7B")
llm_groq = helpers.get_groq_llm("mixtral-8x7b-32768")

# Define the prompt template with a placeholder for the question

system_prompt = "You are an expert wall street options trader. Your trading style is described in the following piece of context. Use it to answer the question at the end. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. Be concise and only respond in the English language."

prompt_template = helpers.create_prompt_template(system_prompt)

context = retriever

# Create an LLMChain to manage interactions with the prompt and model
# llm_chain = LLMChain(
#      prompt=prompt_template,
#      llm=llm_local,
#     )

# Create a chain that uses the Chroma vector store
llm_chain = ConversationalRetrievalChain.from_llm(
     llm = llm_groq,
     chain_type="stuff",
     retriever=context,
     memory=memory,
     return_source_documents=True,
    )

print("Chatbot initialized, ready to chat...")

@app.route('/query', methods=['POST'])
def query():
    # global retrieval_chain
    print("Incoming query...")
    data = request.json
    query = data.get('query')
    print("Query: " + query)
    #local_answer=helpers.search_chroma(query, retriever)
    res = llm_chain.invoke({"question": query})
    print(res, '\n')
    # Process source documents if available
    text_elements = [] # Initialize list to store text elements
    if res['source_documents']:
        for source_idx, source_doc in enumerate(res['source_documents']):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                 {
                    'content': source_doc.page_content,
                    'name': source_name
                }
            )
        #source_names = [text_el['name'] for text_el in text_elements]
    return jsonify(answer=res['answer'], sources=text_elements), 200

if __name__ == '__main__':
    app.run(debug=True)