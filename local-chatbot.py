from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

current_directory = os.getcwd()
print("Current directory:", current_directory)

# Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = LlamaCpp(
    model_path="../../../.cache/lm-studio/models/monal04/llama-2-7b-chat.Q4_0.gguf-GGML/llama-2-7b-chat.Q4_0.gguf",
    n_gpu_layers=40,
    n_batch=512,  # Batch size for model processing
    #verbose=False,  # Enable detailed logging for debugging
)

# Define the prompt template with a placeholder for the question
template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt, llm=llm)

print("Chatbot initialized, ready to chat...")
while True:
    question = input("> ")
    #Did Lady Gaga win an Oscar?
    answer = llm_chain.invoke(question)
    print(answer, '\n')