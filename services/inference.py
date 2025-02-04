import logging
import re
import json
from datetime import datetime
from services.model_loader import tokenizer, model

# Logging setup
logging.basicConfig(level=logging.INFO)

def calculate_age(birth_date_str):
    """Calculates age based on the birthdate."""
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - birth_date.year
        if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
            age -= 1
        return age
    except Exception as e:
        logging.error(f"Error calculating age: {e}")
        return None

def construct_answer(question, context, answer):
    """Construct an answer based on the question and context."""
    question_lower = question.lower()
    
    # Handling questions related to age (example: "How old are you?" with a birthdate in context)
    if "how old" in question_lower and context:
        # Extract birthdate from the context (assuming the format is YYYY-MM-DD)
        match = re.search(r"\d{4}-\d{2}-\d{2}", context)
        if match:
            birth_date_str = match.group(0)
            age = calculate_age(birth_date_str)
            if age is not None:
                return f"I am {age} years old."
            else:
                return "I couldn't calculate the age based on the information provided."
    
    # If the question asks for a specific context-based response
    elif "name" in question_lower:
        return f"My name is {answer}."
    
    # General model-based response construction
    return f"The answer is: {answer}"

def get_answer(question: str, context: str = "Welcome to IntelMind!"):
    """Generates an answer to the question or provides general conversation responses."""
    try:
        # Construct the prompt for the model
        prompt = f"Question: {question}\nContext: {context}\nPlease explain the answer in detail:"
        logging.info(f"Generated prompt: {prompt}")

        # Tokenize inputs
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Forward pass through the model
        outputs = model.generate(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            max_new_tokens=100,  # Increase token limit for a more detailed answer
            temperature=0.7,      # Adjust randomness of the response
            top_k=50             # Control randomness further
        )

        # Decode the generated token IDs to string
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.strip()

        logging.info(f"Generated answer: {answer}")

        # Construct the answer dynamically based on question and context
        constructed_answer = construct_answer(question, context, answer)

        # Return the structured answer in JSON format
        response = {
            "question": question,
            "context": context,
            "answer": constructed_answer if constructed_answer.strip() else "I'm not sure I understood your question. Can you clarify?"
        }

        return json.dumps(response, ensure_ascii=False)  # Return the response as a JSON string

    except Exception as e:
        logging.error(f"Error in generating answer: {e}")
        response = {
            "error": "An error occurred while processing your request. Please try again later."
        }
        return json.dumps(response, ensure_ascii=False)  # Return the error as a JSON string