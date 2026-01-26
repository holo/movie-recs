import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import ast
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional

from dev.db import get_conn

MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

def embed_text(text: str) -> np.ndarray:
    return model.encode(text, normalize_embeddings=True)

def generate_movie_text(movie: dict) -> str:
    parts = [
        f"Title: {movie.get('title', '')}",
        f"Overview: {movie.get('overview', '')}",
        f"Genres: {', '.join(movie.get('genres', []))}",
        f"Keywords: {', '.join(movie.get('keywords', []))}",
        f"Director: {movie.get('director', '')}",
        f"Top Cast: {', '.join(movie.get('top_cast', []))}",
    ]
    return "\n".join(parts)

def fetch_movies_without_embeddings() -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tmdb_id, title, overview, genres, keywords, director, top_cast
                FROM movies
                WHERE embedding IS NULL
            """)
            return cur.fetchall()
    finally:
        conn.close()

def update_movie_embedding(conn, tmdb_id: int, embedding: np.ndarray):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE movies
            SET embedding = %s
            WHERE tmdb_id = %s
        """, (embedding.tolist(), tmdb_id))

def embed_movies(batch_size: int = 64):
    conn = get_conn()
    try:
        rows = fetch_movies_without_embeddings()
        print(f"Found {len(rows)} movies missing embeddings.")

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = [generate_movie_text(dict(r)) for r in batch]

            try:
                embeddings = model.encode(texts, normalize_embeddings=True)
            except Exception as e:
                print(f"Error embedding batch {i}-{i + len(batch)}: {e}")
                continue

            for r, emb in zip(batch, embeddings):
                update_movie_embedding(conn, r["tmdb_id"], emb)

            print(f"Embedded batch {i}-{i + len(batch)} of {len(rows)} movies.")

        conn.commit()
        print("Finished embedding all movies.")
    finally:
        conn.close()

def fetch_user_interactions(user_id: int) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT m.embedding, umi.rating, umi.liked
                FROM user_movie_interactions umi
                JOIN movies m ON umi.tmdb_id = m.tmdb_id
                WHERE umi.user_id = %s AND m.embedding IS NOT NULL
            """, (user_id,))
            return cur.fetchall()
    finally:
        conn.close()

def parse_embedding(embedding) -> Optional[list]:
    if embedding is None:
        return None

    if isinstance(embedding, str):
        try:
            return ast.literal_eval(embedding)
        except Exception:
            return None

    return embedding

def calculate_weighted_embeddings(rows: list[dict]) -> Optional[np.ndarray]:
    embeddings = []
    weights = []

    for r in rows:
        emb = parse_embedding(r["embedding"])
        if emb is None:
            continue

        if r["rating"] is not None:
            weight = (r["rating"] - 2.5) / 2.5
        elif r.get("liked"):
            weight = 0.8
        else:
            continue

        embeddings.append(emb)
        weights.append(weight)

    if not embeddings:
        return None

    embeddings = np.array(embeddings, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)

    return np.average(embeddings, axis=0, weights=weights)

def calculate_residual_taste_vector(rows: list[dict]) -> Optional[np.ndarray]:
    if len(rows) < 3:
        return None
    
    embeddings = []
    ratings = []
    
    for r in rows:
        emb = parse_embedding(r["embedding"])
        if emb is None:
            continue
        
        if r.get("rating") is not None:
            rating = r["rating"]
        elif r.get("liked"):
            rating = 4.0
        else:
            continue
        
        embeddings.append(emb)
        ratings.append(rating)
    
    if len(embeddings) < 3:
        return None
    
    embeddings = np.array(embeddings, dtype=np.float32)
    ratings = np.array(ratings, dtype=np.float32)
    
    if np.any(np.isnan(ratings)) or np.any(np.isinf(ratings)):
        print(f"Warning: Invalid ratings detected, skipping user")
        return None
    
    if np.std(ratings) < 0.01:
        return np.mean(embeddings, axis=0)
    
    centered_ratings = ratings - np.mean(ratings)
    
    taste_vector = np.zeros(768, dtype=np.float32)
    for emb, rating in zip(embeddings, centered_ratings):
        taste_vector += emb * rating
    
    sum_abs_ratings = np.sum(np.abs(centered_ratings))
    if sum_abs_ratings < 1e-8:
        return np.mean(embeddings, axis=0)
    
    taste_vector /= sum_abs_ratings
    
    if np.any(np.isnan(taste_vector)) or np.any(np.isinf(taste_vector)):
        print(f"Warning: Invalid taste vector generated, falling back to average")
        return np.mean(embeddings, axis=0)
    
    return taste_vector

def update_user_taste_vector(conn, user_id: int, taste_vector: np.ndarray):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE users
            SET taste_vector = %s
            WHERE id = %s
        """, (taste_vector.tolist(), user_id))

def update_user_taste_vectors():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users")
            users = cur.fetchall()

        for user in users:
            user_id = user["id"]
            rows = fetch_user_interactions(user_id)

            if not rows:
                print(f"User {user_id} has no embeddings, skipping.")
                continue

            taste_vector = calculate_residual_taste_vector(rows)

            if taste_vector is None:
                print(f"User {user_id} has no weighted embeddings, skipping.")
                continue

            update_user_taste_vector(conn, user_id, taste_vector)
            print(f"Updated taste vector for user {user_id}.")

        conn.commit()
        print("Finished updating all user taste vectors.")
    finally:
        conn.close()

if __name__ == "__main__":
    print("Embedding movies...")
    embed_movies()
    print("Updating user taste vectors...")
    update_user_taste_vectors()
    print("Done")