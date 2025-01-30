# llm-backed-chatbot
goal:
create a chatbot which will help end customer visiting a landing page of a SaaS about product features and guide him to sales. It should support Turkish and English

analysis:
1. choose between RAG,LoRa, fine-tuning/training
   fine-tuning/training is selected. Reason: trained model can be used in further projects.
3. choose llm
4. choose training method/infrastructure
5. choose deployment method/infrastructure
6. choose evaluation method
7. decide on GUI technology stack
   html,css,javascript. Reason: author is familiar with them
9. choose translation api from google,aws,azure


preperation:
1. design
2. prepare training data
3. prepare hld
4. implement solution

flow-runtime: 
user>question-to-chatbot>translator-api>prepare pompt>llm>generated text>post-process>translator-api>answer-to-user
