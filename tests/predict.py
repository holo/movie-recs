import ast
import numpy as np
from typing import Optional, Tuple

from dev.db import get_conn


def parse_vector(v) -> np.ndarray:
    if isinstance(v, str):
        v = ast.literal_eval(v)
    return np.array(v, dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def fetch_user(username: str) -> Optional[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, taste_vector
                FROM users
                WHERE username = %s
            """, (username,))
            row = cur.fetchone()

            if not row or row["taste_vector"] is None:
                return None

            return {
                "id": row["id"],
                "taste_vector": parse_vector(row["taste_vector"])
            }
    finally:
        conn.close()

def fetch_user_rated_movies(user_id: int) -> Tuple[list[np.ndarray], list[float]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT m.embedding, umi.rating
                FROM user_movie_interactions umi
                JOIN movies m ON umi.tmdb_id = m.tmdb_id
                WHERE umi.user_id = %s
                  AND umi.rating IS NOT NULL
                  AND m.embedding IS NOT NULL
            """, (user_id,))
            rows = cur.fetchall()

            embeddings = []
            ratings = []
            for r in rows:
                embeddings.append(parse_vector(r["embedding"]))
                ratings.append(float(r["rating"]))

            return embeddings, ratings
    finally:
        conn.close()

def fetch_unwatched_movies(user_id: int) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tmdb_id, title, embedding
                FROM movies
                WHERE embedding IS NOT NULL
                  AND tmdb_id NOT IN (
                      SELECT tmdb_id
                      FROM user_movie_interactions
                      WHERE user_id = %s
                  )
            """, (user_id,))
            return cur.fetchall()
    finally:
        conn.close()

def fit_option_a_bounds(taste_vector: np.ndarray, watched_embeddings: list[np.ndarray]) -> Tuple[float, float]:
    sims = [cosine_similarity(taste_vector, e) for e in watched_embeddings]
    return min(sims), max(sims)

def predict_option_a(sim: float, sim_min: float, sim_max: float) -> float:
    sim = max(sim_min, min(sim, sim_max))
    return 0.5 + (sim - sim_min) / (sim_max - sim_min) * 4.5

def fit_option_b_model(
    taste_vector: np.ndarray,
    watched_embeddings: list[np.ndarray],
    ratings: list[float]
) -> Tuple[float, float]:
    sims = [cosine_similarity(taste_vector, e) for e in watched_embeddings]
    a, b = np.polyfit(sims, ratings, 1)
    return float(a), float(b)

def predict_option_b(sim: float, a: float, b: float) -> float:
    return float(np.clip(a * sim + b, 0.5, 5.0))

def recommend(username: str, limit: int = 15):
    user = fetch_user(username)
    if not user:
        print("User not found or missing taste vector.")
        return

    taste = user["taste_vector"]
    user_id = user["id"]

    watched_embeddings, ratings = fetch_user_rated_movies(user_id)
    if len(watched_embeddings) < 5:
        print("Not enough ratings to predict reliably.")
        return

    sim_min, sim_max = fit_option_a_bounds(taste, watched_embeddings)
    a, b = fit_option_b_model(taste, watched_embeddings, ratings)

    movies = fetch_unwatched_movies(user_id)
    scored = []

    for m in movies:
        emb = parse_vector(m["embedding"])
        sim = cosine_similarity(taste, emb)

        scored.append({
            "title": m["title"],
            "sim": sim,
            "a": predict_option_a(sim, sim_min, sim_max),
            "b": predict_option_b(sim, a, b),
        })

    scored.sort(key=lambda x: x["b"], reverse=True)

    print(f"\nTop {limit} recommendations:\n")
    for i, m in enumerate(scored[:limit], 1):
        print(
            f"{i}. {m['title']}\n"
            f"   Sim={m['sim']:.3f} | "
            f"A Predicted ★{m['a']:.2f} | "
            f"B Predicted ★{m['b']:.2f}\n"
        )

    print(f"\nBottom {limit} (strongest avoids):\n")
    for i, m in enumerate(scored[-limit:][::-1], 1):
        print(
            f"{i}. {m['title']}\n"
            f"   Sim={m['sim']:.3f} | "
            f"A Predicted ★{m['a']:.2f} | "
            f"B Predicted ★{m['b']:.2f}\n"
        )

if __name__ == "__main__":
    username = input("Username: ").strip()
    recommend(username)