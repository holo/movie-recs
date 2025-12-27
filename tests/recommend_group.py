import ast
import numpy as np

from dev.db import get_conn

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def parse_vector(vector) -> np.ndarray:
    if isinstance(vector, str):
        vector = ast.literal_eval(vector)
    return np.array(vector, dtype=np.float32)

def fetch_user_taste_vectors(usernames: list[str]) -> dict[str, np.ndarray]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT username, taste_vector
                FROM users
                WHERE username = ANY(%s)
                  AND taste_vector IS NOT NULL
            """, (usernames,))
            rows = cur.fetchall()

            return {r["username"]: parse_vector(r["taste_vector"]) for r in rows}
    finally:
        conn.close()

def fetch_seen_movies(usernames: list[str]) -> set[int]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT umi.tmdb_id
                FROM user_movie_interactions umi
                JOIN users u ON u.id = umi.user_id
                WHERE u.username = ANY(%s)
            """, (usernames,))
            return {r["tmdb_id"] for r in cur.fetchall()}
    finally:
        conn.close()

def fetch_candidate_movies(exclude_tmdb_ids: set[int]) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tmdb_id, title, embedding
                FROM movies
                WHERE embedding IS NOT NULL
                  AND NOT (tmdb_id = ANY(%s))
            """, (list(exclude_tmdb_ids),))
            rows = cur.fetchall()

            return [
                {
                    "tmdb_id": r["tmdb_id"],
                    "title": r["title"],
                    "embedding": parse_vector(r["embedding"]),
                }
                for r in rows
            ]
    finally:
        conn.close()

def calculate_group_score(movie_embedding: np.ndarray, taste_vectors: dict[str, np.ndarray]) -> float:
    similarities = [cosine_similarity(user_vec, movie_embedding) for user_vec in taste_vectors.values()]
    return float(np.mean(similarities))

def score_to_rating(score: float) -> float:
    return round(2.5 + score * 2.5, 2)

def recommend_for_group(usernames: list[str], top_n: int = 10) -> list[dict]:
    taste_vectors = fetch_user_taste_vectors(usernames)
    
    if len(taste_vectors) < len(usernames):
        missing = set(usernames) - set(taste_vectors)
        raise RuntimeError(f"Missing taste vectors for users: {missing}")

    seen = fetch_seen_movies(usernames)
    movies = fetch_candidate_movies(seen)

    results = []
    for movie in movies:
        score = calculate_group_score(movie["embedding"], taste_vectors)
        predicted_rating = score_to_rating(score)

        results.append({
            "tmdb_id": movie["tmdb_id"],
            "title": movie["title"],
            "score": score,
            "predicted_rating": predicted_rating,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]

if __name__ == "__main__":
    usernames = input('users (,): ').split(',')
    recs = recommend_for_group(usernames, top_n=10)

    print(f"\nTop recommendations for {', '.join(usernames)}:\n")
    for r in recs:
        print(f"{r['title']} | score={r['score']:.3f} | predicted_rating={r['predicted_rating']}")