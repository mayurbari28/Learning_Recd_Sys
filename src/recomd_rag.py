import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
import json
import openai

#  LOAD DATA
def load_data():
    items = pd.read_csv("data/items.csv")
    interactions = pd.read_csv("data/interactions.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    return items, interactions


#  CHROMA CLIENT (PERSISTENT)
def get_chroma_collection(api_key):
    openai.api_key = api_key

    client = chromadb.Client()

    embed_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-large"
    )

    # Create or load collection
    collection = client.get_or_create_collection(
        name="learning_items",
        embedding_function=embed_func,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


#  INDEX ITEMS INTO CHROMA
def index_items_to_chroma(api_key):
    items, _ = load_data()
    collection = get_chroma_collection(api_key)

    # Prepare documents
    documents = (
        items["title"] + " | category: " + items["category"] +
        " | difficulty: " + items["difficulty"] +
        " | type: " + items["item_type"]
    ).tolist()

    ids = items["item_id"].tolist()

    # Chroma will generate embeddings automatically via embedding_function
    collection.add(documents=documents, ids=ids)

    return collection



#  USER EMBEDDING USING OPENAI
def embed_user_history(api_key, history_text):
    client = openai(api_key=api_key)

    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[history_text]
    )
    return response.data[0].embedding


#  LLM RE-RANKER
def llm_rerank(api_key, user_history_text, candidate_items):
    openai.api_key = api_key

    prompt = f"""
    User learning history:
    {user_history_text}

    Candidate items:
    {candidate_items.to_json(orient="records")}

    Choose the best 5 next learning items based on:
    - topic similarity
    - difficulty progression
    - course → exercise → resource flow
    - prerequisites

    Return JSON array like:
    [
    {{"item_id": "I015", "reason": "Reason"}}
    ]
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return []

class recomd_rag():
    #  MAIN RAG RECOMMENDER USING CHROMA
    def recommend_rag(user_id, api_key):
        items, interactions = load_data()

        # 1. Build / Load Chroma vector index
        collection = index_items_to_chroma(api_key)

        # 2. Get user history
        user_rows = (
            interactions[interactions.user_id == user_id]
            .sort_values("timestamp")
        )

        history_items = user_rows["item_id"].tolist()
        history_titles = user_rows["title"].tolist() if "title" in user_rows else []

        if not history_items:
            return pd.DataFrame(), []

        # 3. Embed user history
        history_text = " ".join(history_titles)
        user_vector = embed_user_history(api_key, history_text)

        # 4. Retrieve top-K from Chroma
        result = collection.query(
            query_embeddings=[user_vector],
            n_results=20
        )

        retrieved_ids = result["ids"][0]
        candidates = items[items["item_id"].isin(retrieved_ids)]

        # 5. LLM re-ranking
        llm_results = llm_rerank(api_key, history_text, candidates)

        if not llm_results:
            return pd.DataFrame(), []

        final_ids = [x["item_id"] for x in llm_results]
        df_final = items[items["item_id"].isin(final_ids)]

        return df_final, llm_results
