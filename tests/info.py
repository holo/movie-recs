import ast
import numpy as np
from typing import Optional

from dev.db import get_conn


def parse_embedding(embedding) -> np.ndarray:
    return np.array(ast.literal_eval(embedding), dtype=np.float32)


def fetch_user_ratings(user_id: int) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT umi.tmdb_id, m.title, m.embedding, umi.rating,
                       m.genres, m.director, m.top_cast, m.keywords
                FROM user_movie_interactions umi
                JOIN movies m ON umi.tmdb_id = m.tmdb_id
                WHERE umi.user_id = %s
                  AND umi.rating IS NOT NULL
                  AND m.embedding IS NOT NULL
            """, (user_id,))
            return cur.fetchall()
    finally:
        conn.close()


def fetch_user_by_username(username: str) -> Optional[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, taste_vector FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            
            if not row:
                return None
            
            return {
                "id": row["id"],
                "taste_vector": np.array(ast.literal_eval(row["taste_vector"]), dtype=np.float32)
            }
    finally:
        conn.close()


def calculate_old_taste_vector(rows: list[dict]) -> np.ndarray:
    embeddings = []
    weights = []

    for r in rows:
        emb = parse_embedding(r["embedding"])
        weight = (r["rating"] - 2.5) / 2.5
        embeddings.append(emb)
        weights.append(weight)

    embeddings = np.array(embeddings, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)

    return np.average(embeddings, axis=0, weights=weights)


def calculate_contrast_taste_vector(rows: list[dict]) -> Optional[np.ndarray]:
    liked_embeddings = []
    disliked_embeddings = []
    
    for r in rows:
        emb = parse_embedding(r["embedding"])
        rating = r["rating"]
        
        if rating >= 3.5:
            weight = (rating - 3.5) / 1.5
            liked_embeddings.append(emb * (1 + weight))
        elif rating <= 2.5:
            weight = (2.5 - rating) / 2.0
            disliked_embeddings.append(emb * (1 + weight))
    
    if not liked_embeddings and not disliked_embeddings:
        return None
    
    liked_vector = np.mean(liked_embeddings, axis=0) if liked_embeddings else np.zeros(768)
    disliked_vector = np.mean(disliked_embeddings, axis=0) if disliked_embeddings else np.zeros(768)
    
    taste_vector = liked_vector - disliked_vector
    
    return taste_vector


def calculate_residual_taste_vector(rows: list[dict]) -> np.ndarray:
    embeddings = []
    ratings = []
    
    for r in rows:
        emb = parse_embedding(r["embedding"])
        embeddings.append(emb)
        ratings.append(r["rating"])
    
    embeddings = np.array(embeddings, dtype=np.float32)
    ratings = np.array(ratings, dtype=np.float32)
    
    centered_ratings = ratings - np.mean(ratings)
    
    taste_vector = np.zeros(768, dtype=np.float32)
    for emb, rating in zip(embeddings, centered_ratings):
        taste_vector += emb * rating
    
    taste_vector /= (np.sum(np.abs(centered_ratings)) + 1e-8)
    
    return taste_vector


def diagnose_taste_vector(username: str):
    user = fetch_user_by_username(username)
    if not user:
        print(f"User '{username}' not found.")
        return
    
    user_id = user["id"]
    current_taste = user["taste_vector"]
    
    rows = fetch_user_ratings(user_id)
    
    print(f"\n{'='*60}")
    print(f"TASTE VECTOR DIAGNOSIS FOR {username}")
    print(f"{'='*60}")
    print(f"Total rated movies: {len(rows)}")
    
    ratings = [r["rating"] for r in rows]
    print(f"\nRating distribution:")
    print(f"  Min: {min(ratings):.1f}, Max: {max(ratings):.1f}")
    print(f"  Mean: {np.mean(ratings):.2f}, Median: {np.median(ratings):.2f}")
    print(f"  Std: {np.std(ratings):.2f}")
    
    print(f"\nCurrent taste vector:")
    print(f"  Norm: {np.linalg.norm(current_taste):.4f}")
    print(f"  Mean: {np.mean(current_taste):.6f}")
    print(f"  Std: {np.std(current_taste):.6f}")
    
    contrast_taste = calculate_contrast_taste_vector(rows)
    residual_taste = calculate_residual_taste_vector(rows)
    
    print(f"\nContrast taste vector (liked - disliked):")
    print(f"  Norm: {np.linalg.norm(contrast_taste):.4f}")
    print(f"  Mean: {np.mean(contrast_taste):.6f}")
    print(f"  Std: {np.std(contrast_taste):.6f}")
    
    print(f"\nResidual taste vector (rating-weighted):")
    print(f"  Norm: {np.linalg.norm(residual_taste):.4f}")
    print(f"  Mean: {np.mean(residual_taste):.6f}")
    print(f"  Std: {np.std(residual_taste):.6f}")
    
    print(f"\n{'='*60}")
    print("DISCRIMINATION TEST - Highest vs Lowest Rated")
    print(f"{'='*60}")
    
    sorted_movies = sorted(rows, key=lambda x: x["rating"], reverse=True)
    top_5 = sorted_movies[:5]
    bottom_5 = sorted_movies[-5:]
    
    def test_vector(taste_vec: np.ndarray, name: str):
        print(f"\n{name}:")
        print("  Top 5 rated movies (should have HIGH scores):")
        for movie in top_5:
            emb = parse_embedding(movie["embedding"])
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            score = np.dot(taste_vec / (np.linalg.norm(taste_vec) + 1e-8), emb_norm)
            print(f"    {movie['title'][:40]:40} | Rating: {movie['rating']:.1f} | Score: {score:.4f}")
        
        print("  Bottom 5 rated movies (should have LOW scores):")
        for movie in bottom_5:
            emb = parse_embedding(movie["embedding"])
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            score = np.dot(taste_vec / (np.linalg.norm(taste_vec) + 1e-8), emb_norm)
            print(f"    {movie['title'][:40]:40} | Rating: {movie['rating']:.1f} | Score: {score:.4f}")
    
    test_vector(current_taste, "Current (Weighted Average)")
    test_vector(contrast_taste, "Contrast (Liked - Disliked)")
    test_vector(residual_taste, "Residual (Rating-Weighted)")
    
    print(f"\n{'='*60}")
    print("SEPARATION METRICS")
    print(f"{'='*60}")
    
    def calc_separation(taste_vec: np.ndarray) -> float:
        top_scores = []
        bottom_scores = []
        taste_norm = taste_vec / (np.linalg.norm(taste_vec) + 1e-8)
        
        for movie in top_5:
            emb = parse_embedding(movie["embedding"])
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            top_scores.append(np.dot(taste_norm, emb_norm))
        
        for movie in bottom_5:
            emb = parse_embedding(movie["embedding"])
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            bottom_scores.append(np.dot(taste_norm, emb_norm))
        
        separation = np.mean(top_scores) - np.mean(bottom_scores)
        return separation
    
    print(f"Current method separation: {calc_separation(current_taste):.4f}")
    print(f"Contrast method separation: {calc_separation(contrast_taste):.4f}")
    print(f"Residual method separation: {calc_separation(residual_taste):.4f}")
    print("\nHigher separation = better discrimination between liked/disliked movies")


if __name__ == "__main__":
    username = input("Username: ").strip()
    diagnose_taste_vector(username)