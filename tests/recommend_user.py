import ast
import numpy as np
from typing import Optional

from dev.db import get_conn

TOP_N = 10

def fetch_user_data(username: str) -> Optional[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, taste_vector FROM users WHERE username = %s", (username,))
            return cur.fetchone()
    finally:
        conn.close()

def fetch_rated_movies(user_id: int) -> set[int]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT tmdb_id FROM user_movie_interactions WHERE user_id = %s", (user_id,))
            return {r["tmdb_id"] for r in cur.fetchall()}
    finally:
        conn.close()

def fetch_movies_with_embeddings():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tmdb_id, title, embedding
                FROM movies
                WHERE embedding IS NOT NULL
            """)
            return cur.fetchall()
    finally:
        conn.close()

def parse_taste_vector(taste_vector) -> np.ndarray:
    return np.array(ast.literal_eval(taste_vector), dtype=np.float32)

def parse_embedding(embedding) -> np.ndarray:
    return np.array(ast.literal_eval(embedding), dtype=np.float32)

def calculate_similarity(taste_vector: np.ndarray, embedding: np.ndarray) -> float:
    return np.dot(taste_vector, embedding) / (np.linalg.norm(taste_vector) * np.linalg.norm(embedding))

def similarity_to_rating(similarity: float) -> float:
    return max(0, min(similarity * 5, 5))

def recommend_movies_for_user(username: str, top_n: int = TOP_N) -> list[tuple]:
    user_data = fetch_user_data(username)
    if not user_data:
        print(f"User '{username}' not found in database.")
        return []

    if user_data["taste_vector"] is None:
        print(f"User '{username}' has no taste vector.")
        return []

    user_id = user_data["id"]
    taste_vector = parse_taste_vector(user_data["taste_vector"])

    rated_movies = fetch_rated_movies(user_id)
    movies = fetch_movies_with_embeddings()

    recommendations = []
    for movie in movies:
        if movie["tmdb_id"] in rated_movies:
            continue

        embedding = parse_embedding(movie["embedding"])
        similarity = calculate_similarity(taste_vector, embedding)
        predicted_rating = similarity_to_rating(similarity)

        recommendations.append((movie["tmdb_id"], movie["title"], predicted_rating, similarity))

    recommendations.sort(key=lambda x: x[2], reverse=True)
    return recommendations[:top_n]

if __name__ == "__main__":
    username = input("Enter username: ").strip()
    recs = recommend_movies_for_user(username)

    print(f"\nTop {len(recs)} predicted ratings for user '{username}':")
    for idx, (tmdb_id, title, rating, sim) in enumerate(recs, 1):
        print(f"{idx}. {title} (TMDB ID: {tmdb_id}) - Predicted Rating: {rating:.2f}, Similarity: {sim:.4f}")