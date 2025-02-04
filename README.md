# llm-backed-chatbot
goal:
create a chatbot which will help end customer visiting a landing page of a SaaS about product features and guide him to sales. It should support Turkish and English

analysis:
1. choose between RAG,LoRa, fine-tuning/training: 
3. choose llm: 
5. choose training method/infrastructure: 
7. choose deployment method/infrastructure
8. choose evaluation method
9. decide on GUI technology stack
   html,css,javascript. Reason: author is familiar with them
10. choose translation api from google,aws,azure


preperation:
1. design
2. prepare training data
3. prepare hld
4. implement solution

flow-runtime: 
user>question-to-chatbot>translator-api>prepare pompt>llm>generated text>post-process>translator-api>answer-to-user

1. BERT + fine-tuning:
   pretraining and training
      colab is used for pretraining and training
      code in preprocess_data.ipynb colab notebook creates trainig data in csv file.
      code in train-bert.ipnnb includes scripts for training the model
      trained model is uplaoded to hugging face as atekrugis/bert_uncased_qa_model
   inference
      virtual machine in azure is used for inference
      endpoint-bert.py calls required libraries and exposes an api using fastapi
   result:
      test with
         curl -X 'POST'   'http://localhost:3456/answer'   -H 'Content-Type: application/json'   -d '{
          "question": "What is the capital of France?",
          "context": "The capital of France is Paris."
        }'
      BERT expect content for Q&A style of text generation. it is not efficient.

2. DeepkSeek R1 7B + RAG
Deployment>RAG>inference endpoint
a. deployment: ollama with cpu only mode used
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-r1:14b
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip -y
pip install -r requirements.txt

3.Llama 3.2 + RAG:

Quantized llama 3.2 self hosted on 32GB memory and 4 cpu

llama-3.2-1b-q5_k_m.gguf is used for inference

/paraphrase-MiniLM-L3-v2 is used for crating embedding vector to query vector store

faiss vector is used for faster response

llama cpp is used for faster response

response time is 1.72 seconds

code: llama_api.py

B. choose chatbot UI

alternatives:

https://deepchat.dev/examples/design


