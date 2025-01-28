import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, GPT2ForQuestionAnswering
import torch
import random
import re

# Set up the router for Text Generation API endpoint
router = APIRouter()

# Pydantic model for request payload
class TextPayload(BaseModel):
    question: str
    context: str = None  # Make context optional

# Pydantic model for response
class TextResponse(BaseModel):
    response: str

# Initialize the tokenizer and model for question answering
try:
    # Load tokenizer and model for question answering
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2ForQuestionAnswering.from_pretrained("openai-community/gpt2")

    logging.info("Model initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing model: {e}")
    raise HTTPException(status_code=500, detail="Error initializing model")

# Default context in case no context is provided in the request
default_context = "Welcome to IntelMind! I am an advanced AI here to assist with processing and understanding professional profiles."

# Helper function to get the answer from the model
def get_answer(question: str, context: str = None) -> str:
    # Use default context if no context is provided or context is empty
    context = context or default_context
    
    # If context is an empty string, ignore and use the default context
    if context == "":
        context = default_context
    
    # Process greetings with dynamic responses
    greetings = ['hi', 'hello', 'hey', 'greetings', 'morning', 'evening', 'yo']
    farewell = ['bye', 'goodbye', 'see you', 'later']
    
    # If it's a greeting, respond with dynamic variations
    if any(greeting in question.lower() for greeting in greetings):
        responses = [
            "Hello! Welcome to IntelMind. How can I assist you today?",
            "Hi there! How can I help you at IntelMind?",
            "Greetings! How can I assist you with your queries today?",
            "Of course! Welcome to IntelMind. What can I help you with?"
        ]
        return random.choice(responses)
    
    # If it's a farewell, respond dynamically
    if any(farewell in question.lower() for farewell in farewell):
        return "Goodbye! It was a pleasure assisting you. Come back anytime!"
    
    # Analyze yes/no questions with dynamic responses
    yes_no_phrases = ['is', 'are', 'can', 'do']
    if any(question.lower().startswith(yn_phrase) for yn_phrase in yes_no_phrases):
        responses = [
            "Yes, let me check that for you.",
            "Certainly! I'll look into that right away.",
            "That's an interesting question! Let me analyze it for you."
        ]
        return random.choice(responses)
    
    # Enhanced name detection using multiple patterns
    # Expanded pattern for recognizing names with different references
    name_patterns = [
        r"my name is (\w+)",  # "my name is John"
        r"i am (\w+)",  # "I am John"
        r"call me (\w+)",  # "Call me Mike"
        r"everyone calls me (\w+)",  # "Everyone calls me Mike"
        r"my full name is (\w+ \w+)",  # "My full name is John Doe"
        r"people usually refer to me as (\w+)",  # "People usually refer to me as Mike"
        r"i'm (\w+)",  # "I'm John"
        r"my [a-zA-Z]+ call me (\w+)",  # Generic pattern "my [person] calls me [name]"
        r"my [a-zA-Z]+ refer to me as (\w+)",  # "My [person] refers to me as [name]"
        r"everyone [a-zA-Z]+ calls me (\w+)",  # "Everyone [person] calls me [name]"
        r"people [a-zA-Z]+ call me (\w+)",  # "People [person] call me [name]"
    ]
    
    # Try each pattern and check if it matches
    for pattern in name_patterns:
        match = re.search(pattern, question.lower())
        if match:
            name = match.group(1).capitalize()
            return f"Nice to meet you, {name}! How can I assist you today?"

    # If the question seems complex or not a greeting, process it with AI
    inputs = tokenizer(question, context, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    # Extracting the answer tokens
    answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Provide a default response if the model doesn't provide an answer
    if not answer.strip():
        return "I'm not sure I understood your question fully. Could you please clarify?"

    return answer

# Endpoint for text generation using GPT-2 (Question Answering)
@router.post("/generate_text/", response_model=TextResponse)
async def generate_text(payload: TextPayload) -> TextResponse:
    try:
        logging.info(f"Received question: {payload.question}")

        # Get the answer based on the provided context or fallback to the default context
        answer = get_answer(payload.question, payload.context)
        
        # If an answer is found, return it, else return a default message
        if answer.strip():
            return TextResponse(response=answer)
        else:
            return TextResponse(response="No specific answer found based on the provided context.")
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Error generating text")