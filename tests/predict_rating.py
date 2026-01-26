import ast
import numpy as np
from typing import Optional, Tuple

from dev.db import get_conn


def fetch_user_taste_vector(username: str) -> Optional[np.ndarray]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT taste_vector FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            
            if not row or row["taste_vector"] is None:
                return None
            
            return np.array(ast.literal_eval(row["taste_vector"]), dtype=np.float32)
    finally:
        conn.close()


def fetch_movie_by_slug(slug: str) -> Optional[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT m.tmdb_id, m.title, m.embedding
                FROM movies m
                JOIN letterboxd_movies lm ON m.tmdb_id = lm.tmdb_id
                WHERE lm.letterboxd_slug = %s
                  AND m.embedding IS NOT NULL
            """, (slug,))
            return cur.fetchone()
    finally:
        conn.close()


def calculate_predicted_rating(
    taste_vector: np.ndarray,
    movie_embedding: np.ndarray
) -> Tuple[float, float]:
    similarity = np.dot(taste_vector, movie_embedding) / (
        np.linalg.norm(taste_vector) * np.linalg.norm(movie_embedding)
    )
    predicted_rating = 2.5 + (similarity * 2.5)
    predicted_rating = max(0.5, min(predicted_rating, 5.0))
    return predicted_rating, similarity


def predict_rating(username: str, letterboxd_slug: str):
    taste_vector = fetch_user_taste_vector(username)
    if taste_vector is None:
        print(f"User '{username}' not found or has no taste vector.")
        return

    movie = fetch_movie_by_slug(letterboxd_slug)
    if not movie:
        print(f"Movie with slug '{letterboxd_slug}' not found in database.")
        return

    movie_embedding = np.array(ast.literal_eval(movie["embedding"]), dtype=np.float32)
    predicted_rating, similarity = calculate_predicted_rating(taste_vector, movie_embedding)

    print(f"\nUser: {username}")
    print(f"Movie: {movie['title']} (TMDB ID: {movie['tmdb_id']})")
    print(f"Predicted Rating: {predicted_rating:.2f} / 5.00")
    print(f"Similarity Score: {similarity:.4f}")


if __name__ == "__main__":
    username = input("Username: ").strip()
    letterboxd_slug = input("Letterboxd slug: ").strip()
    
    predict_rating(username, letterboxd_slug)