import warnings

# Ignora avvisi specifici che possono essere emessi dalle librerie utilizzate
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Importa le librerie necessarie da PyTorch e Hugging Face Transformers
# Hugging face è un progetto che contiene tantimila language model free
import torch
from transformers import logging, BertTokenizer, BertForQuestionAnswering

# Nascondo i log un po' troppo verbosi
logging.set_verbosity_error()

# **** BERT (Bidirectional Encoder Representations from Transformers)
qa_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
qa_model = BertForQuestionAnswering.from_pretrained(qa_model_name)
qa_tokenizer = BertTokenizer.from_pretrained(qa_model_name)

# Configura l'uso della CPU per l'esecuzione dei modelli (invece della GPU)
device = torch.device('cpu') # cuda in caso di GPU
qa_model.to(device)

def answer_question(question, context):
    # encode_plus codifica insieme domanda e contesto
    inputs = qa_tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].tolist()[0]
    print(f"\ninput_ids: {input_ids}")
    
    with torch.no_grad():
        outputs = qa_model(**inputs)
        print(f"\nOutputs: {outputs}")
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    tokens = qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])

    print(f"\nTokens risposta: {tokens}")
    answer = qa_tokenizer.convert_tokens_to_string(tokens)
    
    return answer

# Contesto di esempio su cui fare il Question Answering
context = """
Simon is the father of Lorenzo. Lorenzo is the uncle of Pietro. Pietro's child is Simone.
The red cross is a great thing. I have a dream. The dog is brown. The pen is on the table.
Valentina is "CODESTORM SRL" employe. Simone works with Valentina.
""" 

# Domanda di esempio a cui rispondere
# question = "Which color is the animal?"
# question = "Who is the father of Simone?"
# question = "Who is Mario's wife?"
question = "Which company Simone is working for?"

# Rispondere alla domanda usando il modello di QA
answer = answer_question(question, context)

# Stampa la risposta ottenuta
print(f"Question: {question}")
print(f"Answer: {answer}")
