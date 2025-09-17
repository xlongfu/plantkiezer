import pandas as pd 
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

data = pd.read_csv('data/texas_plant_list_final.csv')

docs = []

for i in range(len(data)):
  t = (
      f"Latin Name: {data.at[i, 'Scientific Name']} | "
      f"Common Name: {data.at[i, 'Common Name']} | "

      # f"Ecoregion: {data.at[i, 'Ecoregion III']} | "
      f"Native Habitat: {data.at[i, 'Native Habitat']} | "

      f"Growth Form: {data.at[i, 'Growth Form']} | "

      f"Bloom Season: {data.at[i, 'Bloom Season']} | "
      f"Bloom Color: {data.at[i, 'Bloom Color']} | "

      # f"Leaf Retention: {data.at[i, 'Leaf Retention']} | "
      f"Lifespan: {data.at[i, 'Lifespan']} | "

      f"Wildlife Benefit: {data.at[i, 'Wildlife Benefit']} | "

      f"Soil: {data.at[i, 'Soil']} | "
      f"Light: {data.at[i, 'Light']} | "
      f"Water: {data.at[i, 'Water']} | "
      
      f"Min Height (cm): {data.at[i, 'Min Height']} | "
      f"Max Height (cm): {data.at[i, 'Max Height']} | "
      f"Min Spread (cm): {data.at[i, 'Min Spread']} | "
      f"Max Spread (cm): {data.at[i, 'Max Spread']} | "

      f"Maintenence: {data.at[i, 'Maintenence']} | "
      f"Comments: {data.at[i, 'Comments']}"
  )
  
  docs.append(t)

uids = data["uid"].astype(str).tolist()

keep_cols = [
    "uid", "Scientific Name", "Common Name", "Native Habitat", "Growth Form",
    "Bloom Season", "Bloom Color", "Lifespan", "Wildlife Benefit", "Soil",
    "Light", "Water", "Min Height", "Max Height", "Min Spread", "Max Spread",
    "Maintenence", "Comments",

    "Leaf Retention", "Price", "Delivery", "Labels"
]

metadatas = data[keep_cols].rename(columns=lambda c: c.replace(" ", "_").lower()).to_dict("records")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

qdrant = Qdrant.from_texts(
    texts=docs,
    embedding=embeddings,
    metadatas=metadatas,
    ids=uids,
    path="vector_stores/plantkiezer",
    collection_name="texas_plants",
)
