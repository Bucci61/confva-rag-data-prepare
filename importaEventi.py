import os
import json
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from urllib.parse import unquote
from datetime import datetime

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

#client = OpenAI(api_key=pcsk_5M9Jm7_EVvNhWLtW7H5VHAToj3JLVGw3YNfnFG6gcPA3RnfZbp2S4EirXSSyq1DDwVs7sp)
#pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


INDEX_NAME = "confindustria-eventi"

# crea indice se non esiste
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(INDEX_NAME)
# elimina TUTTI i vettori dall'indice
#index.delete(delete_all=True)

def decode(s: str) -> str:
    """Decodifica stringhe URL-encoded come 'abc%20def'."""
    if not isinstance(s, str):
        return ""
    return unquote(s)

def build_text(item):
    def decode(s):
        return s.replace("%20", " ")

    parts = [
        decode(item.get("titolo", "")),
        decode(item.get("data", "")),
        decode(item.get("descrizione", "")),
        decode(item.get("AreaInteresse", "")),
        decode(item.get("Settori", "")),
        decode(item.get("Tags", ""))
    ]
    return "\n".join([p for p in parts if p])

def embed(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding


def embed_text(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def parse_date(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.strftime("%Y%m%d"))

def parse_date_str(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y-%m-%d")


def ingest_json(path):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    vectors = []

    for item in items:
        if item.get("unid") == "-1":
            continue

        doc_id = item["unid"]
        text = build_text(item)
        vector = embed(text)

        embedding = embed_text(text)

        print(parse_date_str(item.get("data")))

        vectors.append({
            "id": doc_id,
            "values": embedding,
            "metadata": {
                "unid": item.get("unid"),
                "titolo": decode(item.get("titolo")),
                "data": parse_date_str(item.get("data")),
                "text": text,
                "settori": decode(item.get("Settori")),
                "areainteresse": decode(item.get("AreaInteresse")),
                "tags": decode(item.get("Tags")),
                # 🔥 INDICATORE DELLA FONTE
                "source": "confindustria_varese_eventi"
            }
        })




    index.upsert(vectors)
    print(f"{len(vectors)} documenti indicizzati.")

if __name__ == "__main__":
    ingest_json("eventiConfindustria.json")

