import warnings
import fitz 
import os
import torch
from transformers import logging, BertTokenizer, BertForQuestionAnswering, BertModel
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Nascondo i log un po' troppo verbosi
logging.set_verbosity_error()

# Configurazione del modello BERT per question answering
qa_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
qa_model = BertForQuestionAnswering.from_pretrained(qa_model_name)
qa_tokenizer = BertTokenizer.from_pretrained(qa_model_name)

# Modello BERT per ottenere gli embedding
embed_model_name = 'bert-base-uncased'
embed_model = BertModel.from_pretrained(embed_model_name)
embed_tokenizer = BertTokenizer.from_pretrained(embed_model_name)

# Configura l'uso della CPU per l'esecuzione dei modelli (invece della GPU)
device = torch.device('cpu')  # 'cuda' in caso di GPU
qa_model.to(device)
embed_model.to(device)

# Funzione per ottenere gli embedding di una frase
def get_embeddings(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

# Connessione al database e creazione della tabella per gli embedding
conn = psycopg2.connect(
    dbname="vectordb",
    user="postgres",
    password="testpassword",
    host="localhost",
    port="5432"
)

register_vector(conn)

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    text TEXT,
    embedding VECTOR(768)  -- Dimensione dell'embedding BERT
);
DELETE FROM embeddings;
""")
conn.commit()

# Salvataggio degli embedding nel database
def save_embeddings(texts):
    # print('save_embeddings says: ' + texts)
    if isinstance(texts, str):
        texts = [texts]
    embeddings = [get_embeddings(text) for text in texts]
    data = [(text, embedding) for text, embedding in zip(texts, embeddings)]
    execute_values(cur, "INSERT INTO embeddings (text, embedding) VALUES %s", data)
    conn.commit()

# Esempio di frasi da salvare nel database
# texts = [
#      "Simone is 42 years old.",
#      "Simone eyes are brown colored.",
#      "Simone have two legs!",
#      "Mario have three arms.",
#      "Camilla is 7 years old and Alessandro is 9 years old",
# ]
# save_embeddings(texts)


def transform_string_to_list(text):
    # Split the text by newlines and strip whitespace from each line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines

path = os.path.abspath(os.getcwd())
fullpath = path + '/prova.pdf'
doc = fitz.open(fullpath) 
testo = ""
for page in doc: 
   testo += page.get_text()
   newtesto = testo.replace("\n", " ").replace("\r", "").strip()
   struttura = newtesto.split()
#    print('>>>' + testo)ß
   save_embeddings(transform_string_to_list(testo))
         

# print("TESTI: <<<" + newtesto + " >>>")


def answer_question(question, context):
    # encode_plus codifica insieme domanda e contesto
    inputs = qa_tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].tolist()[0]
    with torch.no_grad():
        outputs = qa_model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    tokens = qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    answer = qa_tokenizer.convert_tokens_to_string(tokens)
    return answer

# Funzione per rispondere a una domanda usando gli embedding nel database
def answer_question_with_db(question):
    print(f"Rispondo alla domanda passando dal database")
    question_embedding = get_embeddings(question)
    cur.execute("SELECT text, embedding <=> %s AS distance FROM embeddings ORDER BY distance LIMIT 1", (question_embedding,))
    result = cur.fetchone()
    if result:
        print(f"Ho trovato un risultato: {result[0]}")
        answer = answer_question(question, result[0])
        return answer
    else:
        print(f"Non ho trovato risultati")
        return "No relevant context found in the database."

# Esempio di domanda a cui rispondere
question = "How old is Simone?"
question = "Which is the colour of Simone's eyes?"
question = "How many legs have Simone?"
question = "How many arms have Mario?"
question = "How old is Camilla?"
question = "How old is Alessandro?"

# Rispondere alla domanda usando il modello di QA e il database
answer = answer_question_with_db(question)

# Stampa la risposta ottenuta
print(f"Question: {question}")
print(f"Answer: {answer}")

# Chiusura del cursore e della connessione
cur.close()
conn.close()
