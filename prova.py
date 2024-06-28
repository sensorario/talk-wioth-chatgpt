import warnings

# Ignora avvisi specifici che possono essere emessi dalle librerie utilizzate
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Importa le librerie necessarie da PyTorch e Hugging Face Transformers
# Hugging face è un progetto che contiene tantimila language model free
import torch
from transformers import logging, GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertForQuestionAnswering

# Nascondo i log
logging.set_verbosity_error()

# Qui viene caricato il modello preaddestratop gpt2 ed il tokenizer
# Che cosa è un modello pre-addestrato?
    # E' un tipo di intelligenza artificiale  addestrata su un amio set di dati.
# Che cosa è un tokenizer
    # Traduce un testo in dati in modo che possa essere digeribile da un modello
    # Pre processa i dati
# Perchè un tokenizer è associato ad un modello?
    # Un tokenizer può essere utilizzato con un modello che usa la stessa architettura 
# Carica il tokenizer associato al modello GPT-2
gpt2_model_name = 'gpt2'
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

# Nome del modello BERT pre-addestrato per il Question Answering
# Carica il modello pre-addestrato BERT per il Question Answering
# Carica il tokenizer associato al modello BERT per il Question Answering
# **** BERT (Bidirectional Encoder Representations from Transformers)
# **** BERT (Bidirectional Encoder Representations from Transformers)
qa_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
qa_model = BertForQuestionAnswering.from_pretrained(qa_model_name)
qa_tokenizer = BertTokenizer.from_pretrained(qa_model_name)

# Configura l'uso della CPU per l'esecuzione dei modelli (invece della GPU)
device = torch.device('cpu')
gpt2_model.to(device)
qa_model.to(device)

# Funzione per ottenere gli embeddings GPT-2 di un testo
def get_gpt2_embeddings(text):
    # Codifica il testo in input IDs utilizzando il tokenizer GPT-2
    input_ids = gpt2_tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Disabilita il calcolo del gradiente per l'ottenimento degli embeddings
    with torch.no_grad():
        # Ottiene gli output del modello, inclusi gli stati nascosti (hidden states)
        outputs = gpt2_model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    
    # Prende l'ultimo stato nascosto (hidden state) dal modello
    # Calcola la media degli embeddings sull'asse delle sequenze
    last_hidden_state = hidden_states[-1]
    mean_embedding = last_hidden_state.mean(dim=1)
    
    return mean_embedding

def answer_question(question, context):
    inputs = qa_tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].tolist()[0]
    # print(f"input_ids: {input_ids}")
    
    with torch.no_grad():
        outputs = qa_model(**inputs)
        # print(f"Outputs: {outputs}")
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    tokens = qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    # print(f"Tokens: {tokens}")
    answer = qa_tokenizer.convert_tokens_to_string(tokens)
    
    return answer

# Contesto di esempio su cui fare il Question Answering
context = """
Simon is the father of Lorenzo. Lorenzo is the uncle of Pietro. Pietro's child is Simone.
The red cross is a great thing. I have a dream. The dog is brown. The pen is on the table.
Valentina is CODESTORM employe. Simone works with Valentina.
""" 

# Domanda di esempio a cui rispondere
question = "Which color is the animal?"
question = "Who is the father of Simone?"
question = "Who is Mario's wife?"
question = "Which company Simone is working for?"

# Ottenere gli embeddings GPT-2 del contesto (non necessario per il modello di QA, ma mostrato come esempio)
# gpt2_embedding = get_gpt2_embeddings(context)
# print(f"GPT-2 Embedding: {gpt2_embedding}")

# Rispondere alla domanda usando il modello di QA
answer = answer_question(question, context)

# Stampa la risposta ottenuta
print(f"Question: {question}")
print(f"Answer: {answer}")
