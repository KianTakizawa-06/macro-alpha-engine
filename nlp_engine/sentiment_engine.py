import sqlite3
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# 1. Load the pre-trained FinBERT model
print("Loading FinBERT Model (This might take a moment)...")
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# 2. Document Chunking Algorithm
def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# 3. The Scoring Engine
def analyze_sentiment(text):
    if not text or len(text.strip()) == 0:
        return 0.0, 0.0, 0.0

    chunks = chunk_text(text)
    hawkish_total, dovish_total, neutral_total = 0, 0, 0
    
    for chunk in chunks:
        # Tokenize and run through the neural network
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        
        # Convert logits to probabilities using Softmax
        probs = F.softmax(outputs.logits, dim=1).detach().numpy()[0]
        
        # FinBERT-tone outputs: [Neutral, Positive (Hawkish), Negative (Dovish)]
        neutral_total += probs[0]
        hawkish_total += probs[1]
        dovish_total += probs[2]
        
    # Calculate the average score across all chunks
    num_chunks = len(chunks)
    return hawkish_total / num_chunks, dovish_total / num_chunks, neutral_total / num_chunks

# 4. Process the Database
def score_database():
    conn = sqlite3.connect('macro_engine.db')
    cursor = conn.cursor()
    
    # Add new columns to our database to store the scores
    try:
        cursor.execute("ALTER TABLE text_data ADD COLUMN hawkish_score REAL")
        cursor.execute("ALTER TABLE text_data ADD COLUMN dovish_score REAL")
        cursor.execute("ALTER TABLE text_data ADD COLUMN neutral_score REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass # Columns already exist
        
    # Select documents that haven't been scored yet
    cursor.execute("SELECT id, date, source, text_content FROM text_data WHERE hawkish_score IS NULL")
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} unscored documents. Starting analysis...")
    
    for row in rows:
        doc_id, date, source, text_content = row
        print(f"Scoring {source} document from {date}...")
        
        hawk_score, dove_score, neut_score = analyze_sentiment(text_content)
        
        # Update the database with the new scores
        cursor.execute('''
            UPDATE text_data 
            SET hawkish_score = ?, dovish_score = ?, neutral_score = ?
            WHERE id = ?
        ''', (hawk_score, dove_score, neut_score, doc_id))
        
    conn.commit()
    conn.close()
    print("Sentiment analysis complete!")

if __name__ == "__main__":
    score_database()

    import sqlite3
import struct

conn = sqlite3.connect('macro_engine.db')
cursor = conn.cursor()

cursor.execute("SELECT id, hawkish_score, dovish_score, neutral_score FROM text_data WHERE hawkish_score IS NOT NULL")
rows = cursor.fetchall()

fixed = 0
for row in rows:
    doc_id, hawk, dove, neut = row
    if isinstance(hawk, bytes):
        hawk = struct.unpack('<f', hawk)[0]
        dove = struct.unpack('<f', dove)[0]
        neut = struct.unpack('<f', neut)[0] if isinstance(neut, bytes) else neut
        cursor.execute(
            "UPDATE text_data SET hawkish_score = ?, dovish_score = ?, neutral_score = ? WHERE id = ?",
            (float(hawk), float(dove), float(neut), doc_id)
        )
        fixed += 1

conn.commit()
conn.close()
print(f"Repaired {fixed} rows")