import os
import PyPDF2
import bs4
import pandas as pd
from sqlalchemy import create_engine, inspect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_groq import ChatGroq
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from flask import jsonify

CHROMA_DATA_PATH = "chroma/"

#Parse PDF file to text
def pdf_to_txt(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    return pdf_text

def docs_to_chroma(docs, isRebuild):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = LlamaCppEmbeddings(model_path=os.getenv("MODEL_NOMIC_EMBED_TEXT"))

    if isRebuild:
        #rebuild the chroma db
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings,persist_directory=CHROMA_DATA_PATH)
    else:
        vectorstore = Chroma(embedding_function=embeddings,persist_directory=CHROMA_DATA_PATH)

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore

    return retriever

def site_to_doc(url):
    # Load, chunk and index the contents of the blog.
    bs_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs_strainer},
    )
    docs = loader.load()

    return docs

def txt_to_doc(file_path):
    loader = TextLoader(file_path)
    # loader = TextLoader("./data/NOTES.txt")
    docs = loader.load()
    return docs

def get_local_llm(model):
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Load the LlamaCpp language model, adjust GPU usage based on your hardware
    llm_local = LlamaCpp(
        #Get local model path defined in .env
        model_path=os.getenv(model),
        n_gpu_layers=0, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
        temperature=0.2,
        top_k=40,
        repeat_penalty=1.1,
        top_p=0.95,
        n_threads=4, # The number of CPU threads to use, tailor to your system and the resulting performance
        n_batch=512,  # Batch size for model processing
        n_ctx=2048, # The max sequence length to use - note that longer sequence lengths require much more resources
        callback_manager=callback_manager,
        verbose=False,  # Verbose is required to pass to the callback manager
    )

    return llm_local

def get_groq_llm(model):
    llm_groq = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            #model_name='llama2-70b-4096' 
            model_name=model,
            temperature=0
    )

    return llm_groq

def create_prompt_template(system_prompt):
    system_template_str = "[INST]"
    system_template_str += system_prompt
    system_template_str += " Context: {context}[/INST]"

    new_system_prompt = SystemMessagePromptTemplate(
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
    messages = [new_system_prompt, human_prompt]

    prompt_template = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=messages,
    )

    return prompt_template

def search_chroma(question, retriever):
    local_answer=retriever.similarity_search(question, k=3)
    print(local_answer, '\n')

# Create an LLMChain to manage interactions with the prompt and model
def create_llm_chain(prompt_template, llm):
    llm_chain = LLMChain(
        prompt=prompt_template,
        llm = llm
        )

def run_local_app(llm_chain):
    while True:
        question = input("> ")
        context = context
        answer = llm_chain.invoke({"context": context, "question": question})
        print(answer, '\n')

def get_sources(res):
    # Process source documents if available
    sources = [] # Initialize list to store text elements
    if res['context']:
        for source_idx, source_doc in enumerate(res['context']):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            print(source_doc)
            sources.append(
                 {
                    'page_content': source_doc.page_content,
                    'name': source_name,                    
                    'metadata': source_doc.metadata
                }
            )
    return sources

def csv_to_sql_db(full_file_path):
    file = os.path.basename(full_file_path)
    file_name, file_extension = os.path.splitext(file)
    db_path = "sql/" + file_name + ".db"
    db_path = f"sqlite:///{db_path}"
    new_db = create_engine(db_path)
    df = pd.read_csv(full_file_path)
    df.to_sql(file_name, new_db, index=False)
    print("All csv files are saved into the sql database.")