import logging
import numpy as np
from services.model_loader import tokenizer, onnx_session  # Importing from model_loader

# Logging setup
logging.basicConfig(level=logging.INFO)

def get_answer(question: str, context: str = "Welcome to IntelMind!"):
    """Uses the ONNX model to generate an answer for the given question."""
    try:
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        logging.info(f"Generated prompt: {prompt}")

        # Tokenize inputs
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

        input_ids = inputs["input_ids"].cpu().numpy()  # Moving input to CPU and converting to numpy
        attention_mask = inputs["attention_mask"].cpu().numpy()

        ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        # Run the ONNX model
        ort_outputs = onnx_session.run(None, ort_inputs)
        generated_ids = ort_outputs[0]  # Assuming first output is the generated token IDs

        # Decode the generated token IDs to string
        answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        logging.info(f"Generated answer: {answer}")

        # Return the answer or a fallback response if it's empty
        return answer if answer.strip() else "I'm not sure I understood your question. Can you clarify?"

    except Exception as e:
        logging.error(f"Error in generating answer: {e}")
        return "An error occurred while processing your request. Please try again later."