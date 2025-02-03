import logging
import re
from services.model_loader import tokenizer, model

# Logging setup
logging.basicConfig(level=logging.INFO)

# Multilingual greetings list (you can expand it as needed)
greetings_multilingual = {
    'english': ['hi', 'hello', 'hey', 'good morning', 'good evening', 'howdy', 'hola'],
    'spanish': ['hola', 'buenos días', 'buenas tardes', 'buenas noches'],
    'french': ['salut', 'bonjour', 'bonsoir'],
    'german': ['hallo', 'guten morgen', 'guten abend'],
    'italian': ['ciao', 'buongiorno', 'buonasera'],
    'portuguese': ['olá', 'bom dia', 'boa tarde', 'boa noite'],
    'chinese': ['你好', '早安', '晚上好'],
    'japanese': ['こんにちは', 'おはよう', 'こんばんは'],
    'arabic': ['مرحبًا', 'صباح الخير', 'مساء الخير'],
    # Add more languages here...
}

def detect_greeting(statement):
    """Detect if the statement is a greeting in any language."""
    statement = statement.lower()
    for lang, greetings in greetings_multilingual.items():
        if any(greeting in statement for greeting in greetings):
            return lang  # Return the language of the greeting
    return None  # No greeting detected

def get_answer(question: str, context: str = "Welcome to IntelMind!"):
    """Generates an answer to the question or provides general conversation responses."""
    try:
        # Detect greeting language
        greeting_language = detect_greeting(question)
        if greeting_language:
            return f"Hi! Welcome to IntelMind. How can I assist you today? ({greeting_language.capitalize()} greeting detected)"
        
        # Check if the statement is asking for information about IntelMind
        elif "about" in question.lower() or "who are you" in question.lower() or "what is intel mind" in question.lower():
            return "IntelMind is an AI-powered chatbot designed to assist with information and conversation. How can I help you today?"

        # For other questions, use the model to generate an answer
        else:
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

            # Return the answer or a fallback response if it's empty
            return answer if answer.strip() else "I'm not sure I understood your question. Can you clarify?"

    except Exception as e:
        logging.error(f"Error in generating answer: {e}")
        return "An error occurred while processing your request. Please try again later."