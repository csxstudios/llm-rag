import pandas as pd
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
#from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.chains import LLMChain

df = pd.read_csv('./data/oscars.csv')
df = df.loc[df['year_ceremony'] == 2023]
df = df.dropna(subset=['film'])
df.loc[:, 'category'] = df['category'].str.lower()
df.loc[:, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' to win the award'
df.loc[df['winner'] == False, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' but did not win'               

client = chromadb.Client()
collection = client.get_or_create_collection("oscars-2023")

docs = df["text"].tolist() 

vector_db = Chroma(
    "langchain_store",
    collection
)

# Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = LlamaCpp(
    model_path="../../../.cache/lm-studio/models/monal04/llama-2-7b-chat.Q4_0.gguf-GGML/llama-2-7b-chat.Q4_0.gguf",
    n_gpu_layers=40,
    n_batch=512,  # Batch size for model processing
    #verbose=False,  # Enable detailed logging for debugging
)

# Define the prompt template with a placeholder for the question
system_template_str = """You are a helpful AI assistant that can answer questions on Oscar 2023 awards. Answer based on the context provided. If you cannot find the correct answerm, say I don't know. Be concise and just include the response.

{context}
"""

system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=system_template_str,
    )
)

human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [system_prompt, human_prompt]

prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

context = vector_db.as_retriever(k=3)

print("Chatbot initialized, ready to chat...")
while True:
    #context="I am healthy!"
    question = input("> ")
    context = context
    answer = llm_chain.invoke({"context": context, "question": question})
    print(answer, '\n')