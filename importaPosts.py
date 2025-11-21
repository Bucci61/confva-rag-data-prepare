import os
import json
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from urllib.parse import unquote
from datetime import datetime

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


INDEX_NAME = "confindustria-posts"

# Se l'indice esiste, lo cancello
#if INDEX_NAME in [idx["name"] for idx in pc.list_indexes()]:
#    pc.delete_index(INDEX_NAME)

# Ora lo ricreo con la dimensione giusta (1536 per text-embedding-3-small)
#pc.create_index(
#    name=INDEX_NAME,
#    dimension=1536,   # <-- dimensione corretta per text-embedding-3-small
#    metric="cosine",
#    spec=ServerlessSpec(cloud="aws", region="us-east-1")
#)


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
        decode(item.get("title", "")),
        decode(item.get("date", "")),
        decode(item.get("url", "")),
        decode(item.get("category", "")),
        decode(item.get("categoryfull", "")),
        decode(item.get("content", ""))
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

def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """
    Divide il testo in blocchi di max_chars caratteri.
    Puoi regolare max_chars in base alla lunghezza media dei tuoi contenuti.
    """
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]


def ingest_json(path):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    vectors = []
    n = 0
    for item in items:
        if item.get("unid") == "-1":
            continue
        n += 1
        print(str(n))

        doc_id = item["unid"]
        text = build_text(item)

        # 🔹 Spezza il testo in chunk
        chunks = chunk_text(text, max_chars=2000)

        for idx, chunk in enumerate(chunks):
            embedding = embed_text(chunk)

            vectors.append({
                "id": f"{doc_id}_chunk{idx}",
                "values": embedding,
                "metadata": {
                    "unid": item.get("unid"),
                    "title": decode(item.get("title")),
                    "date": item.get("date"),
                    "text": chunk,  # solo il pezzo corrente
                    "category": decode(item.get("category")),
                    "categoryfull": decode(item.get("categoryfull")),
                    "url": decode(item.get("url")),
                    "source": "confindustria_varese_post",
                    "chunk_index": idx,
                    "chunk_total": len(chunks)
                }
            })

    # 🔹 Upsert di tutti i vettori
    index.upsert(vectors)
    print(f"{len(vectors)} vettori indicizzati (da {n} documenti).")


if __name__ == "__main__":
    ingest_json("postsConfindustria.json")

