import os
import json
from datetime import datetime
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from urllib.parse import unquote

# ======================================================================
#  INIT OpenAI + Pinecone
# ======================================================================

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# I TRE INDICI USATI NEL RAG
INDEXES = {
    "posts": "confindustria-posts",
    "news": "confindustria-news",
    "eventi": "confindustria-eventi"
}

# CREA GLI INDICI SE NON ESISTONO
for idx in INDEXES.values():
    if idx not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=idx,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

# Istanziare gli indici Pinecone
pinecone_indexes = {label: pc.Index(name) for label, name in INDEXES.items()}



# ======================================================================
#  FUNZIONI DI UTILITÀ
# ======================================================================

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

def embed_text(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def chunk_text(text: str, max_chars: int = 2000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]




# ======================================================================
#  🔎 RICERCA MULTI-INDICE + RICOMPOSIZIONE
# ======================================================================

def search_and_recompose(query: str, top_k: int = 5):
    """
    Cerca la query nei tre indici:
    - confindustria-posts
    - confindustria-news
    - confindustria-eventi
    Unisce i risultati e ricompone i documenti.
    """

    # 1️⃣ Embedding query
    query_embedding = embed_text(query)

    all_matches = []

    # 2️⃣ Query su ciascun indice
    for label, index in pinecone_indexes.items():
        try:
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            for m in results["matches"]:
                m["metadata"]["__source_index"] = label

            all_matches.extend(results["matches"])

        except Exception as e:
            print(f"⚠️ Errore nell'indice {label}: {e}")

    # 3️⃣ Ordina per punteggio
    all_matches = sorted(all_matches, key=lambda x: x["score"], reverse=True)

    # 4️⃣ Raggruppamento chunk per documento
    docs = {}

    for match in all_matches:
        metadata = match["metadata"]

        unid = metadata.get("unid")
        source = metadata.get("__source_index")

        unique_key = f"{source}_{unid}"

        if unique_key not in docs:
            docs[unique_key] = {
                "unid": unid,
                "source": source,
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "date": metadata.get("date"),
                "category": metadata.get("category"),
                "chunks": {}
            }

        chunk_index = metadata.get("chunk_index")
        docs[unique_key]["chunks"][chunk_index] = metadata.get("text")

    # 5️⃣ Ricomposizione documento completo
    recomposed_docs = []

    for key, doc in docs.items():
        ordered = [doc["chunks"][i] for i in sorted(doc["chunks"].keys())]
        full_text = "\n".join(ordered)

        recomposed_docs.append({
            "unid": doc["unid"],
            "source": doc["source"],
            "title": doc["title"],
            "url": doc["url"],
            "date": doc["date"],
            "category": doc["category"],
            "content": full_text
        })

    return recomposed_docs



# ======================================================================
#  PROMPTING GPT
# ======================================================================

def prompt_with_index(query: str, top_k: int = 5):

    docs = search_and_recompose(query, top_k=top_k)

    context = "\n\n".join([
        f"🔹 SORGENTE: {doc['source']}\n"
        f"TITOLO: {doc['title']}\nURL: {doc['url']}\n"
        f"DATA: {doc['date']}\nCATEGORIA: {doc['category']}\n"
        f"CONTENUTO:\n{doc['content']}"
        for doc in docs
    ])


    date_now = datetime.today()
    date_now_formatted = date_now.strftime("%d/%m/%Y")

    prompt = f"""
Sei un assistente intelligente che risponde usando i documenti di Confindustria Varese provenienti da:
- confindustria-posts
- confindustria-news
- confindustria-eventi

Domanda utente: {query}

Documenti rilevanti:
{context}

Rispondi in modo chiaro, conciso e indica sempre la SORGENTE delle informazioni. Qualora sia indicato, specifica sempre la URL Non cercare mai informazioni oltre quelle contentute nel RAG. Quando sono richieste informazioni future tieni conto che la data odierna è {date_now_formatted}. 
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    #print("\n--- RISPOSTA ---\n")
    print(response.choices[0].message.content)



# ======================================================================
#  MAIN
# ======================================================================

if __name__ == "__main__":

    print("🔎 Ricerca multi-indice Confindustria")
    print("Scrivi la tua domanda (exit per uscire)\n")

    while True:
        query = input("👉 Inserisci il prompt: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        try:
            prompt_with_index(query, top_k=3)
        except Exception as e:
            print("⚠️ Errore:", e)
