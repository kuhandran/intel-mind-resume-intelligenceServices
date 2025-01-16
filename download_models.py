from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForTokenClassification, BertTokenizer
import torch

def save_models():
    # Ensure PyTorch is installed and log its version
    print(f"PyTorch version: {torch.__version__}")

    # Saving GPT-2 model and tokenizer
    gpt2_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    gpt2_model.save_pretrained('./models/distilgpt2')
    gpt2_tokenizer.save_pretrained('./models/distilgpt2')
    print("GPT-2 model and tokenizer saved in './models/distilgpt2'")
    
    # Saving BERT model and tokenizer
    bert_model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
    bert_tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
    bert_model.save_pretrained('./models/dbmdz/bert-large-cased-finetuned-conll03-english')
    bert_tokenizer.save_pretrained('./models/dbmdz/bert-large-cased-finetuned-conll03-english')
    print("BERT model and tokenizer saved in './models/dbmdz/bert-large-cased-finetuned-conll03-english'")

if __name__ == "__main__":
    save_models()