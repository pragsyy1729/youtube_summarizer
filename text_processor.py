from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load pre-trained model and tokenizer
model_name = "t5-base"  # You can use other variants like "t5-small" or "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def answer_question(context, question):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer