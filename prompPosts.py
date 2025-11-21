import os
import json
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from urllib.parse import unquote
from datetime import datetime

# 🔑 Client OpenAI e Pinecone
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "confindustria-posts"

# ✅ Crea indice solo se non esiste
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,   # text-embedding-3-small produce vettori di 1536 dimensioni
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# -------------------------------
# Funzioni di utilità
# -------------------------------

def decode(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unquote(s)

def build_text(item):
    def decode_local(s):
        return s.replace("%20", " ")
    parts = [
        decode_local(item.get("title", "")),
        decode_local(item.get("date", "")),
        decode_local(item.get("url", "")),
        decode_local(item.get("category", "")),
        decode_local(item.get("categoryfull", "")),
        decode_local(item.get("content", ""))
    ]
    return "\n".join([p for p in parts if p])

def embed_text(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# -------------------------------
# Ingestione documenti
# -------------------------------

def ingest_json(path):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    vectors = []
    n = 0
    for item in items:
        if item.get("unid") == "-1":
            continue
        n += 1
        print(f"Ingestione documento {n}")

        doc_id = item["unid"]
        text = build_text(item)

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
                    "text": chunk,
                    "category": decode(item.get("category")),
                    "categoryfull": decode(item.get("categoryfull")),
                    "url": decode(item.get("url")),
                    "source": "confindustria_varese_post",
                    "chunk_index": idx,
                    "chunk_total": len(chunks)
                }
            })

    index.upsert(vectors)
    print(f"{len(vectors)} vettori indicizzati (da {n} documenti).")

# -------------------------------
# Ricerca + ricomposizione
# -------------------------------

def search_and_recompose(query: str, top_k: int = 5):
    # 1. Embedding della query
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # 2. Query su Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # 3. Raggruppa chunk per documento
    docs = {}
    for match in results["matches"]:
        metadata = match["metadata"]
        unid = metadata.get("unid")
        if unid not in docs:
            docs[unid] = {
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "date": metadata.get("date"),
                "category": metadata.get("category"),
                "chunks": {}
            }
        docs[unid]["chunks"][metadata.get("chunk_index")] = metadata.get("text")

    # 4. Ricomposizione
    recomposed_docs = []
    for unid, doc in docs.items():
        ordered_chunks = [doc["chunks"][i] for i in sorted(doc["chunks"].keys())]
        full_text = "\n".join(ordered_chunks)
        recomposed_docs.append({
            "unid": unid,
            "title": doc["title"],
            "url": doc["url"],
            "date": doc["date"],
            "category": doc["category"],
            "content": full_text
        })

    return recomposed_docs

# -------------------------------
# Prompting con i documenti
# -------------------------------

def prompt_with_index(query: str, top_k: int = 5):
    docs = search_and_recompose(query, top_k=top_k)

    # Costruisci un contesto con i documenti trovati
    context = "\n\n".join([
        f"TITOLO: {doc['title']}\nURL: {doc['url']}\nDATA: {doc['date']}\nCATEGORIA: {doc['category']}\nCONTENUTO:\n{doc['content']}"
        for doc in docs
    ])

    # Prompt finale
    prompt = f"""Sei un assistente che risponde basandosi sui documenti di Confindustria Varese.
Domanda utente: {query}

Documenti rilevanti:
{context}

Rispondi in modo chiaro e sintetico, citando i documenti se utile. Utilizza sempre solo i documenti del RAG.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n--- RISPOSTA ---\n")
    print(response.choices[0].message.content)

# -------------------------------
# Esecuzione
# -------------------------------

if __name__ == "__main__":
    # Prima ingestione (solo la prima volta, poi commenta)
    # ingest_json("postsConfindustria.json")

    print("🔎 Prompting su indice Confindustria Varese")
    print("Scrivi la tua domanda (digita 'exit' per uscire)\n")

    while True:
        query = input("👉 Inserisci il prompt: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Uscita dal programma.")
            break

        try:
            prompt_with_index(query, top_k=3)
        except Exception as e:
            print(f"⚠️ Errore durante l'elaborazione: {e}")




#if __name__ == "__main__":
    # Prima ingestione (solo la prima volta)
    # ingest_json("postsConfindustria.json")

    # Poi query + prompting
    #prompt_with_index("Descrivi in breve il contenuto dei post di confindustria varese", top_k=3)
    #prompt_with_index("innovazione digitale nelle PMI", top_k=3)
    #prompt_with_index("Cerca i post più interessanti sulla intelligenza artificiale. Dammi informazioni, commenti e url per accedere ai post originali da te individuati", top_k=3)
    #prompt_with_index("Come è la situazione industriale della provincia di varese nel 2025 e quali sono le prospettive future", top_k=3)
