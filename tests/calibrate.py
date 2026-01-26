import ast
import numpy as np
from typing import Optional, Tuple

from dev.db import get_conn

def parse_embedding(embedding) -> np.ndarray:
    return np.array(ast.literal_eval(embedding), dtype=np.float32)

def calculate_content_score(taste_vector: np.ndarray, embedding: np.ndarray) -> float:
    taste_norm = taste_vector / (np.linalg.norm(taste_vector) + 1e-8)
    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    return np.dot(taste_norm, emb_norm)

def fetch_user_data(user_id: int) -> Optional[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT taste_vector FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            
            if not row or row["taste_vector"] is None:
                return None
            
            return {
                "taste_vector": np.array(ast.literal_eval(row["taste_vector"]), dtype=np.float32)
            }
    finally:
        conn.close()

def fetch_watched_movies(user_id: int) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT umi.tmdb_id, m.title, m.embedding, umi.rating, 
                       m.genres, m.keywords, m.director, m.top_cast
                FROM user_movie_interactions umi
                JOIN movies m ON umi.tmdb_id = m.tmdb_id
                WHERE umi.user_id = %s
                  AND umi.rating IS NOT NULL
                  AND m.embedding IS NOT NULL
            """, (user_id,))
            return cur.fetchall()
    finally:
        conn.close()

def fetch_all_users() -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT u.id, u.username
                FROM users u
                JOIN user_movie_interactions umi ON u.id = umi.user_id
                WHERE umi.rating IS NOT NULL
            """)
            return cur.fetchall()
    finally:
        conn.close()

def save_user_calibration(user_id: int, metadata_weight: float):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE users 
                ADD COLUMN IF NOT EXISTS metadata_weight REAL DEFAULT 0.3
            """)
            
            cur.execute("""
                UPDATE users 
                SET metadata_weight = %s 
                WHERE id = %s
            """, (metadata_weight, user_id))
        
        conn.commit()
    finally:
        conn.close()

def load_user_calibration(user_id: int) -> float:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT metadata_weight FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            
            if row and row["metadata_weight"] is not None:
                return row["metadata_weight"]
            
            return 0.3
    finally:
        conn.close()

def learn_keyword_preferences(watched_movies: list[dict]) -> dict:
    keyword_ratings = {}
    
    for movie in watched_movies:
        rating = movie["rating"]
        for kw in (movie["keywords"] or []):
            keyword_ratings.setdefault(kw, []).append(rating)
    
    return {
        kw: np.mean(ratings) 
        for kw, ratings in keyword_ratings.items() 
        if len(ratings) >= 2
    }

def learn_director_preferences(watched_movies: list[dict]) -> dict:
    director_ratings = {}
    
    for movie in watched_movies:
        rating = movie["rating"]
        director = movie["director"]
        if director:
            director_ratings.setdefault(director, []).append(rating)
    
    return {
        director: np.mean(ratings) 
        for director, ratings in director_ratings.items()
    }

def learn_cast_preferences(watched_movies: list[dict]) -> dict:
    cast_ratings = {}
    
    for movie in watched_movies:
        rating = movie["rating"]
        for actor in (movie["top_cast"] or []):
            cast_ratings.setdefault(actor, []).append(rating)
    
    return {
        actor: np.mean(ratings) 
        for actor, ratings in cast_ratings.items() 
        if len(ratings) >= 2
    }

def calculate_metadata_boost(
    movie: dict,
    keyword_scores: dict,
    director_scores: dict,
    cast_scores: dict,
    user_mean: float
) -> float:
    boost = 0.0
    feature_count = 0
    
    keywords = movie.get("keywords") or []
    if keywords:
        keyword_boosts = [
            keyword_scores.get(kw, user_mean) - user_mean 
            for kw in keywords if kw in keyword_scores
        ]
        if keyword_boosts:
            boost += np.mean(keyword_boosts)
            feature_count += 1
    
    director = movie.get("director")
    if director and director in director_scores:
        boost += (director_scores[director] - user_mean)
        feature_count += 1
    
    cast = movie.get("top_cast") or []
    if cast:
        cast_boosts = [
            cast_scores.get(actor, user_mean) - user_mean 
            for actor in cast if actor in cast_scores
        ]
        if cast_boosts:
            boost += np.mean(cast_boosts)
            feature_count += 1
    
    if feature_count > 0:
        return boost / feature_count
    
    return 0.0

def predict_with_params(
    taste_vector: np.ndarray,
    movie: dict,
    keyword_scores: dict,
    director_scores: dict,
    cast_scores: dict,
    user_mean: float,
    min_score: float,
    max_score: float,
    min_rating: float,
    max_rating: float,
    metadata_weight: float
) -> float:
    embedding = parse_embedding(movie["embedding"])
    content_score = calculate_content_score(taste_vector, embedding)
    
    metadata_boost = calculate_metadata_boost(
        movie, keyword_scores, director_scores, cast_scores, user_mean
    )
    
    enhanced_score = content_score + metadata_weight * metadata_boost
    
    if max_score - min_score < 1e-6:
        return (min_rating + max_rating) / 2
    
    normalized = (enhanced_score - min_score) / (max_score - min_score)
    predicted = min_rating + normalized * (max_rating - min_rating)
    
    if enhanced_score < min_score:
        predicted = min_rating - (min_score - enhanced_score) * (max_rating - min_rating) / (max_score - min_score)
    elif enhanced_score > max_score:
        predicted = max_rating + (enhanced_score - max_score) * (max_rating - min_rating) / (max_score - min_score)
    
    return np.clip(predicted, 0.5, 5.0)

def calibrate_user_params(
    user_id: int,
    taste_vector: np.ndarray,
    watched_movies: list[dict]
) -> Tuple[float, float]:
    if len(watched_movies) < 10:
        return 0.3, float('inf')
    
    keyword_scores = learn_keyword_preferences(watched_movies)
    director_scores = learn_director_preferences(watched_movies)
    cast_scores = learn_cast_preferences(watched_movies)
    user_mean = np.mean([m["rating"] for m in watched_movies])
    
    temp_scores = []
    for movie in watched_movies:
        embedding = parse_embedding(movie["embedding"])
        content_score = calculate_content_score(taste_vector, embedding)
        metadata_boost = calculate_metadata_boost(
            movie, keyword_scores, director_scores, cast_scores, user_mean
        )
        temp_scores.append(content_score + 0.3 * metadata_boost)
    
    min_score = np.min(temp_scores)
    max_score = np.max(temp_scores)
    min_rating = np.min([m["rating"] for m in watched_movies])
    max_rating = np.max([m["rating"] for m in watched_movies])
    
    weight_candidates = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    best_weight = 0.3
    best_mae = float('inf')
    
    for metadata_weight in weight_candidates:
        errors = []
        
        for movie in watched_movies:
            predicted = predict_with_params(
                taste_vector, movie,
                keyword_scores, director_scores, cast_scores, user_mean,
                min_score, max_score, min_rating, max_rating,
                metadata_weight
            )
            
            actual = movie["rating"]
            errors.append(abs(predicted - actual))
        
        mae = np.mean(errors)
        
        if mae < best_mae:
            best_mae = mae
            best_weight = metadata_weight
    
    return best_weight, best_mae

def calibrate_all_users():
    users = fetch_all_users()
    
    print(f"{'='*80}")
    print(f"PER-USER HYPERPARAMETER CALIBRATION")
    print(f"{'='*80}\n")
    
    results = []
    
    for user in users:
        user_id = user["id"]
        username = user["username"]
        
        user_data = fetch_user_data(user_id)
        if not user_data:
            print(f"{username}: No taste vector, skipping.")
            continue
        
        watched_movies = fetch_watched_movies(user_id)
        if len(watched_movies) < 10:
            print(f"{username}: Only {len(watched_movies)} ratings, skipping (need 10+).")
            continue
        
        print(f"Calibrating {username}...")
        
        best_weight, best_mae = calibrate_user_params(
            user_id, user_data["taste_vector"], watched_movies
        )
        
        print(f"  ✓ Best metadata_weight: {best_weight:.2f} (MAE: {best_mae:.3f})\n")
        
        results.append({
            "username": username,
            "metadata_weight": best_weight,
            "mae": best_mae,
            "num_ratings": len(watched_movies)
        })
    
    print(f"\n{'='*80}")
    print(f"CALIBRATION SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Username':<20} {'Ratings':>8} {'Weight':>8} {'MAE':>8}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    
    for r in sorted(results, key=lambda x: x["mae"]):
        print(f"{r['username']:<20} {r['num_ratings']:>8} {r['metadata_weight']:>8.2f} {r['mae']:>8.3f}")
    
    avg_mae = np.mean([r["mae"] for r in results])
    print(f"\nAverage MAE: {avg_mae:.3f}")
    
    weights = [r["metadata_weight"] for r in results]
    print(f"\nMetadata weight distribution:")
    print(f"  Min:    {min(weights):.2f}")
    print(f"  Max:    {max(weights):.2f}")
    print(f"  Median: {np.median(weights):.2f}")
    print(f"  Mean:   {np.mean(weights):.2f}")

if __name__ == "__main__":
    calibrate_all_users()