import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

CHROMA_DATA_PATH = "chroma/"

#Parse TXT file to text
def TxtToText(filePath):
    with open('./data/NOTES.txt', 'r') as file:
        txt_text = file.read()
    
    return txt_text

#Parse PDF file to text
def PdfToText(filePath):
    pdf = PyPDF2.PdfReader(filePath)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    return pdf_text

#Text to local chroma vector db
def TextToChroma(text, isRebuild):
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    embeddings = LlamaCppEmbeddings(model_path=os.getenv("MODEL_NOMIC_EMBED_TEXT"))
    #embeddings = OllamaEmbeddings(model="llama2:7b")

    if isRebuild:
        print("Building local chroma vector db...")
        #rebuild the chroma db
        retriever = Chroma.from_texts(
            texts, embeddings, metadatas=metadatas, persist_directory=CHROMA_DATA_PATH
        )
        print("Completed chroma vector db")

    retriever = Chroma(embedding_function=embeddings,persist_directory=CHROMA_DATA_PATH)

    return retriever

def GetLocalLLM(model):
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Load the LlamaCpp language model, adjust GPU usage based on your hardware
    llm_local = LlamaCpp(
        #Get local model path defined in .env
        model_path=os.getenv(model),
        #n_gpu_layers=40,
        temperature=0.2,
        top_k=40,
        repeat_penalty=1.1,
        top_p=0.95,
        n_threads=4,
        n_batch=512,  # Batch size for model processing
        n_ctx=2048,
        callback_manager=callback_manager,
        verbose=False,  # Verbose is required to pass to the callback manager
    )

    return llm_local

def GetGroqLLM(model):
    llm_groq = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            #model_name='llama2-70b-4096' 
            model_name=model
    )

    return llm_groq

def CreatePromptTemplate(systemPrompt):
    system_template_str = "[INST]"
    system_template_str += systemPrompt
    system_template_str += " Context: {context}[/INST]"

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

    return prompt_template

def SearchChroma(question, retriever):
    local_answer=retriever.similarity_search(question, k=3)
    print(local_answer, '\n')

def RunAppLocal(llm_chain):
    while True:
        question = input("> ")
        context = context
        answer = llm_chain.invoke({"context": context, "question": question})
        print(answer, '\n')