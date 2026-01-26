import ast
import numpy as np
from typing import Optional
from dev.db import get_conn

def parse_embedding(embedding) -> np.ndarray:
    return np.array(ast.literal_eval(embedding), dtype=np.float32)

def calculate_content_score(taste_vector: np.ndarray, embedding: np.ndarray) -> float:
    taste_norm = taste_vector / (np.linalg.norm(taste_vector) + 1e-8)
    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    return np.dot(taste_norm, emb_norm)

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

def learn_keyword_preferences(watched_movies: list[dict]) -> dict:
    keyword_ratings = {}

    for movie in watched_movies:
        rating = movie["rating"]
        for kw in movie["keywords"] or []:
            keyword_ratings.setdefault(kw, []).append(rating)

    keyword_scores = {}
    for kw, ratings in keyword_ratings.items():
        if len(ratings) >= 2:
            mean = np.mean(ratings)
            count = len(ratings)
            weight = min(1.0, count / 5.0)
            keyword_scores[kw] = (mean, weight)

    return keyword_scores

def learn_director_preferences(watched_movies: list[dict]) -> dict:
    director_ratings = {}

    for movie in watched_movies:
        rating = movie["rating"]
        director = movie["director"]
        if director:
            director_ratings.setdefault(director, []).append(rating)

    director_scores = {}
    for director, ratings in director_ratings.items():
        mean = np.mean(ratings)
        count = len(ratings)
        weight = min(1.0, count / 3.0)
        director_scores[director] = (mean, weight)

    return director_scores

def learn_cast_preferences(watched_movies: list[dict]) -> dict:
    cast_ratings = {}

    for movie in watched_movies:
        rating = movie["rating"]
        for actor in movie["top_cast"] or []:
            cast_ratings.setdefault(actor, []).append(rating)

    cast_scores = {}
    for actor, ratings in cast_ratings.items():
        if len(ratings) >= 2:
            mean = np.mean(ratings)
            count = len(ratings)
            weight = min(1.0, count / 4.0)
            cast_scores[actor] = (mean, weight)

    return cast_scores

def calculate_metadata_boost(
    movie: dict,
    keyword_scores: dict,
    director_scores: dict,
    cast_scores: dict,
    user_mean: float
) -> float:
    deltas = []
    
    for kw in movie.get("keywords") or []:
        if kw in keyword_scores:
            mean, weight = keyword_scores[kw]
            delta = (mean - user_mean) * weight
            delta = float(np.clip(delta, -1.0, 1.0))
            deltas.append(delta)

    director = movie.get("director")
    if director and director in director_scores:
        mean, weight = director_scores[director]
        delta = (mean - user_mean) * weight
        delta = float(np.clip(delta, -1.0, 1.0))
        deltas.append(delta)

    for actor in movie.get("top_cast") or []:
        if actor in cast_scores:
            mean, weight = cast_scores[actor]
            delta = (mean - user_mean) * weight
            delta = float(np.clip(delta, -1.0, 1.0))
            deltas.append(delta)

    if not deltas:
        return 0.0

    mean_delta = float(np.mean(deltas))
    feature_count = len(deltas)

    confidence = min(1.0, feature_count / 10.0)
    meta_weight = 0.1 + 0.2 * confidence

    boost = mean_delta * meta_weight
    return boost

def fit_enhanced_mapping(user_id: int, taste_vector: np.ndarray) -> Optional[dict]:
    watched_movies = fetch_watched_movies(user_id)
    
    if len(watched_movies) < 5:
        return None
    
    keyword_scores = learn_keyword_preferences(watched_movies)
    director_scores = learn_director_preferences(watched_movies)
    cast_scores = learn_cast_preferences(watched_movies)
    
    user_mean = np.mean([m["rating"] for m in watched_movies])
    
    enhanced_scores = []
    ratings = []
    
    for movie in watched_movies:
        embedding = parse_embedding(movie["embedding"])
        
        content_score = calculate_content_score(taste_vector, embedding)
        
        metadata_boost = calculate_metadata_boost(
            movie, keyword_scores, director_scores, cast_scores, user_mean
        )
        
        enhanced_score = content_score + metadata_boost
        
        enhanced_scores.append(enhanced_score)
        ratings.append(movie["rating"])
    
    scores = np.array(enhanced_scores)
    ratings = np.array(ratings)
    
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_ratings = ratings[sorted_indices]
    
    return {
        "scores": sorted_scores,
        "ratings": sorted_ratings,
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "min_rating": np.min(ratings),
        "max_rating": np.max(ratings),
        "keyword_scores": keyword_scores,
        "director_scores": director_scores,
        "cast_scores": cast_scores,
        "user_mean": user_mean
    }

def predict_rating_enhanced(content_score: float, movie: dict, mapping_info: Optional[dict]) -> float:
    if mapping_info is None:
        return 2.75 + content_score * 2.25
    
    metadata_boost = calculate_metadata_boost(
        movie,
        mapping_info["keyword_scores"],
        mapping_info["director_scores"],
        mapping_info["cast_scores"],
        mapping_info["user_mean"]
    )
    
    enhanced_score = content_score + metadata_boost
    
    min_score = mapping_info["min_score"]
    max_score = mapping_info["max_score"]
    min_rating = mapping_info["min_rating"]
    max_rating = mapping_info["max_rating"]
    
    if max_score - min_score < 1e-6:
        return (min_rating + max_rating) / 2
    
    normalized = (enhanced_score - min_score) / (max_score - min_score)
    predicted = min_rating + normalized * (max_rating - min_rating)
    
    if enhanced_score < min_score:
        predicted = min_rating - (min_score - enhanced_score) * (max_rating - min_rating) / (max_score - min_score)
    elif enhanced_score > max_score:
        predicted = max_rating + (enhanced_score - max_score) * (max_rating - min_rating) / (max_score - min_score)
    
    return np.clip(predicted, 0.5, 5.0)

def evaluate_user(user_id: int, username: str) -> Optional[dict]:
    user_data = fetch_user_data(user_id)
    if not user_data:
        return None
    
    taste_vector = user_data["taste_vector"]
    watched_movies = fetch_watched_movies(user_id)
    
    if len(watched_movies) < 5:
        return None
    
    mapping_info = fit_enhanced_mapping(user_id, taste_vector)
    if not mapping_info:
        return None
    
    predictions = []
    errors = []
    absolute_errors = []
    
    for movie in watched_movies:
        embedding = parse_embedding(movie["embedding"])
        content_score = calculate_content_score(taste_vector, embedding)
        predicted_rating = predict_rating_enhanced(content_score, movie, mapping_info)
        actual_rating = movie["rating"]
        
        error = predicted_rating - actual_rating
        abs_error = abs(error)
        
        predictions.append({
            "title": movie["title"],
            "predicted": predicted_rating,
            "actual": actual_rating,
            "error": error,
            "abs_error": abs_error
        })
        
        errors.append(error)
        absolute_errors.append(abs_error)
    
    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    
    predictions.sort(key=lambda x: x["abs_error"], reverse=True)
    worst_5 = predictions[:5]
    
    predictions.sort(key=lambda x: x["abs_error"])
    best_5 = predictions[:5]
    
    return {
        "username": username,
        "num_ratings": len(watched_movies),
        "mae": mae,
        "rmse": rmse,
        "worst_predictions": worst_5,
        "best_predictions": best_5,
        "rating_mean": np.mean([m["rating"] for m in watched_movies]),
        "rating_std": np.std([m["rating"] for m in watched_movies])
    }

def evaluate_all_users():
    users = fetch_all_users()
    
    print(f"{'='*80}")
    print(f"MULTI-USER RECOMMENDER EVALUATION")
    print(f"Mean-based metadata boost (improved version)")
    print(f"{'='*80}\n")
    
    results = []
    
    for user in users:
        user_id = user["id"]
        username = user["username"]
        
        print(f"Evaluating user: {username}...")
        result = evaluate_user(user_id, username)
        
        if result:
            results.append(result)
            print(f"  ✓ MAE: {result['mae']:.3f}, RMSE: {result['rmse']:.3f} ({result['num_ratings']} ratings)\n")
        else:
            print(f"  ✗ Insufficient data\n")
    
    if not results:
        print("No users with sufficient data to evaluate.")
        return
    
    print(f"\n{'='*80}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*80}")
    
    overall_mae = np.mean([r["mae"] for r in results])
    overall_rmse = np.mean([r["rmse"] for r in results])
    
    print(f"\nUsers evaluated: {len(results)}")
    print(f"Average MAE across all users: {overall_mae:.3f}")
    print(f"Average RMSE across all users: {overall_rmse:.3f}")
    
    print(f"\n{'='*80}")
    print(f"PER-USER BREAKDOWN")
    print(f"{'='*80}\n")
    
    results.sort(key=lambda x: x["mae"])
    
    print(f"{'Username':<20} {'Ratings':>8} {'MAE':>8} {'RMSE':>8} {'Avg Rating':>12} {'Std Dev':>10}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*10}")
    
    for r in results:
        print(f"{r['username']:<20} {r['num_ratings']:>8} {r['mae']:>8.3f} {r['rmse']:>8.3f} {r['rating_mean']:>12.2f} {r['rating_std']:>10.2f}")
    
    print(f"\n{'='*80}")
    print(f"WORST PREDICTIONS ACROSS ALL USERS")
    print(f"{'='*80}\n")
    
    all_worst = []
    for r in results:
        for pred in r["worst_predictions"][:3]:
            all_worst.append({
                "username": r["username"],
                "title": pred["title"],
                "predicted": pred["predicted"],
                "actual": pred["actual"],
                "error": pred["abs_error"]
            })
    
    all_worst.sort(key=lambda x: x["error"], reverse=True)
    
    for i, pred in enumerate(all_worst[:15], 1):
        print(f"{i:2d}. [{pred['username']:<12}] {pred['title'][:45]:45}")
        print(f"    Predicted: {pred['predicted']:.2f} | Actual: {pred['actual']:.1f} | Error: {pred['error']:.2f}\n")
    
    print(f"{'='*80}")
    print(f"MAE DISTRIBUTION ANALYSIS")
    print(f"{'='*80}\n")
    
    maes = [r["mae"] for r in results]
    print(f"Best MAE:    {min(maes):.3f}")
    print(f"Worst MAE:   {max(maes):.3f}")
    print(f"Median MAE:  {np.median(maes):.3f}")
    print(f"25th %ile:   {np.percentile(maes, 25):.3f}")
    print(f"75th %ile:   {np.percentile(maes, 75):.3f}")
    
    excellent = [r for r in results if r["mae"] < 0.40]
    good = [r for r in results if 0.40 <= r["mae"] < 0.60]
    fair = [r for r in results if 0.60 <= r["mae"] < 0.80]
    poor = [r for r in results if r["mae"] >= 0.80]
    
    print(f"\nPerformance Categories:")
    print(f"  Excellent (MAE < 0.40): {len(excellent)} users")
    print(f"  Good (0.40-0.60):       {len(good)} users")
    print(f"  Fair (0.60-0.80):       {len(fair)} users")
    print(f"  Poor (≥0.80):           {len(poor)} users")
    
    if poor:
        print(f"\n  Users with poor performance (MAE ≥ 0.80):")
        for r in poor:
            print(f"    • {r['username']}: {r['mae']:.3f} MAE ({r['num_ratings']} ratings)")

if __name__ == "__main__":
    evaluate_all_users()