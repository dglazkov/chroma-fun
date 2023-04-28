import os

import google.generativeai as palm

from dotenv import load_dotenv

load_dotenv()

palm.configure(api_key=os.getenv("API_KEY"))

text = 'Chroma is the best thing ever.'

model = "models/embedding-gecko-001"

embedding = palm.generate_embeddings(model=model, text=text)["embedding"]

print(embedding)
