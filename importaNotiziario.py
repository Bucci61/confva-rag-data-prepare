import os
import json
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from urllib.parse import unquote
from datetime import datetime

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "confindustria-news"

# crea indice se non esiste
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


# -----------------------------------------------------------
# UTILITY
# -----------------------------------------------------------

def decode(s: str) -> str:
    """Decodifica stringhe URL-encoded."""
    if not isinstance(s, str):
        return ""
    return unquote(s)


def build_text(item):
    """Costruisce il testo da indicizzare concatenando i campi utili."""
    fields = [
        item.get("title", ""),
        item.get("subject", ""),
        item.get("content", ""),
        item.get("circolareinbreve", ""),
        item.get("settore", ""),
        item.get("areatematica", ""),
        item.get("interesse", "")
    ]
    return "\n".join([decode(f) for f in fields if f])


def embed_text(text: str) -> list:
    """Richiede l'embedding a OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Spezzetta il testo in blocchi di dimensione massima."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]


def upsert_in_batches(index, vectors, batch_size=50):
    """Esegue l'upsert in batch per evitare limiti Pinecone."""
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(batch)
        print(f"Upsertati {len(batch)} vettori (batch {i//batch_size + 1})")


# -----------------------------------------------------------
# INGESTIONE JSON
# -----------------------------------------------------------

def ingest_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    vectors = []
    n = 0

    for item in items:
        if item.get("unid") == "-1":
            continue

        n += 1
        print(f"Documento {n}: {item.get('title','(senza titolo)')}")

        doc_id = item["unid"]
        full_text = build_text(item)

        # Spezza in chunk
        chunks = chunk_text(full_text, max_chars=2000)

        for idx, chunk in enumerate(chunks):
            embedding = embed_text(chunk)

            vectors.append({
                "id": f"{doc_id}_chunk{idx}",
                "values": embedding,
                "metadata": {
                    "unid": item.get("unid"),
                    "title": decode(item.get("title")),
                    "date": item.get("date"),
                    "settore": decode(item.get("settore")),
                    "areatematica": decode(item.get("areatematica")),
                    "interesse": decode(item.get("interesse")),
                    "text": chunk,               # SOLO il chunk → metadata leggera!
                    "source": "notiziario",
                    "chunk_index": idx,
                    "chunk_total": len(chunks)
                }
            })

    # Upsert sicuro e in batch
    upsert_in_batches(index, vectors, batch_size=40)

    print(f"\n✔️ Indicizzati {len(vectors)} vettori da {n} documenti.")


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

if __name__ == "__main__":
    ingest_json("notiziarioConfindustria.json")
