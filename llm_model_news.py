
import numpy as np 
import pandas as pd
news = pd.read_csv('labelled_newscatcher_dataset.csv', sep=';')
MAX_NEWS = 1000
DOCUMENT="title"
TOPIC="topic"

#news = pd.read_csv('/kaggle/input/bbc-news/bbc_news.csv')
#MAX_NEWS = 1000
#DOCUMENT="description"
#TOPIC="title"

#news = pd.read_csv('/kaggle/input/mit-ai-news-published-till-2023/articles.csv')
#MAX_NEWS = 100
#DOCUMENT="Article Body"
#TOPIC="Article Header"

news["id"] = news.index
news.head()
#Because it is just a course we select a small portion of News.
subset_news = news.head(MAX_NEWS)
import chromadb
from chromadb.config import Settings
#OLD VERSION
#settings_chroma = Settings(chroma_db_impl="duckdb+parquet", 
#                          persist_directory='./input')
#chroma_client = chromadb.Client(settings_chroma)

#NEW VERSION => 0.40
chroma_client = chromadb.PersistentClient(path="/path/to/persist/directory")
collection_name = "news_collection"
if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
        chroma_client.delete_collection(name=collection_name)

collection = chroma_client.create_collection(name=collection_name)
    

collection.add(
    documents=subset_news[DOCUMENT].tolist(),
    metadatas=[{TOPIC: topic} for topic in subset_news[TOPIC].tolist()],
    ids=[f"id{x}" for x in range(MAX_NEWS)],
)
results = collection.query(query_texts=["laptop"], n_results=10 )

print(results)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

getado = collection.get(ids="id141", 
                       include=["documents", "embeddings"])

word_vectors = getado["embeddings"]
word_list = getado["documents"]
word_vectors
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "databricks/dolly-v2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
lm_model = AutoModelForCausalLM.from_pretrained(model_id)


pipe = pipeline(
    "text-generation",
    model=lm_model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    device_map="auto",
)
question = "Can I buy a Toshiba laptop?"
context = " ".join([f"#{str(i)}" for i in results["documents"][0]])
#context = context[0:5120]
prompt_template = f"Relevant context: {context}\n\n The user's question: {question}"
prompt_template
lm_response = pipe(prompt_template)
print(lm_response[0]["generated_text"])


# Assuming the model and necessary functions are defined above,
# the following function will handle chat input for inference.

# Code for saving the model using pickle
import pickle

# Assuming 'model' is your trained model variable
# Replace 'model' with the actual variable name of your model
model_artifact_path = 'model_artifact_llm_news.pkl'  # Path where the model artifact will be saved
with open(model_artifact_path, 'wb') as file:
    pickle.dump(lm_model, file)

print("model saved in to file")