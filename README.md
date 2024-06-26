# llm-rag
Implementing RAG with Chroma and Llama 2
- https://youtu.be/c02FEBTukwM?si=7ENO804d1q3j0xnx
- Explore how word embeddings work and the role of Chroma in improving the precision of text generation (chroma.ipynb)

Building a Local Chatbot w/ Langchain and Llama
- local-chatbot.py
- https://medium.com/@weidagang/hello-llm-building-a-local-chatbot-with-langchain-and-llama2-3a4449fc4c03

Building an LLM RAG Chatbot With Langchain
- local-chatbot-rag-v1.py
- https://realpython.com/build-llm-rag-chatbot-with-langchain/

Building a RAG Chatbot with PDFs
- local-chatbot-rag-v2.py
- https://github.com/sudarshan-koirala/rag-chat-with-pdf/
- https://console.groq.com/docs/showcase-applications

Chain w/ Chat History Returning Sources
- local-chatbot-rag-v3.py
- https://python.langchain.com/v0.1/docs/use_cases/question_answering/chat_history/
- https://python.langchain.com/v0.1/docs/use_cases/question_answering/sources/

Get currenty directory
```bash
import os

current_directory = os.getcwd()
print("Current directory:", current_directory)

```

## Other tutorials
- https://youtu.be/cMJWC-csdK4?si=5kvA64ctvfqW2go6
- https://youtu.be/TOeAe8KB68E?si=Gy3PD30U4QcZFRrF

#Windows powershell commands for CUDA
- Install Cuda 12.2 development toolkit
- Install Microsoft Build Tools 2019 and choose C++ development for CMake
```bash
$env:CMAKE_ARGS="-DLLAMA_CUDA=on"
$env:FORCE_CMAKE="1"
$env:CUDA_VISIBLE_DEVICES="0"
python -m pip install llama-cpp-python --force-reinstall --no-cache-dir --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cuda122
```