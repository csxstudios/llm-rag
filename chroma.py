import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

df = pd.read_csv('./data/oscars.csv')

df = df.loc[df['year_ceremony'] == 2023]

df = df.dropna(subset=['film'])
df.loc[:, 'category'] = df['category'].str.lower()

df.loc[df['winner'] == True, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' and won.'
df.loc[df['winner'] == False, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ', but did not win.'

client = chromadb.Client()
collection = client.get_or_create_collection("oscars-2023")

docs=df["text"].tolist()

docs[0]

ids = [str(x) for x in df.index.tolist()]

collection.add(
    documents=docs,
    ids=ids
)

collection.peek()

results  = collection.query(
    query_texts=["lady gaga"],
    n_results=10
)
print(results['documents'])