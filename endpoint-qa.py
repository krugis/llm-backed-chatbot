from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Initialize FastAPI
app = FastAPI()

# Load the model and tokenizer
model = BertForQuestionAnswering.from_pretrained("/home/bert-endpoint/model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Pydantic model to parse the request body
class QuestionContext(BaseModel):
    question: str
    context: str

# Function for answering questions
def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get start and end logits
    outputs = model(input_ids, attention_mask=attention_mask)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits

    # Get the most likely start and end positions
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Get the answer tokens
    answer_ids = input_ids[0][start_index:end_index+1]
    
    # Decode the answer tokens into the answer string
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    
    # Ensure the answer doesn't include the question (cleanup)
    answer = answer.strip().replace(question, "").strip()
    
    # In case question is still part of the answer, clean it further
    if answer.lower().startswith(question.lower()):
        answer = answer[len(question):].strip()

    return answer

# API endpoint to get answers
@app.post("/answer")
def get_answer(item: QuestionContext):
    answer = answer_question(item.question, item.context)
    return {"answer": answer}
