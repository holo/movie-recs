import ast
import numpy as np
from typing import Optional

from dev.db import get_conn

def parse_embedding(embedding) -> np.ndarray:
    if isinstance(embedding, str):
        embedding = ast.literal_eval(embedding)
    return np.array(embedding, dtype=np.float32)

def calculate_content_score(taste_vector: np.ndarray, embedding: np.ndarray) -> float:
    taste_norm = taste_vector / (np.linalg.norm(taste_vector) + 1e-8)
    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    return np.dot(taste_norm, emb_norm)

def fetch_user_data(username: str) -> Optional[dict]:
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
                "username": username,
                "taste_vector": parse_embedding(row["taste_vector"])
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

def fetch_group_watched_movies(usernames: list[str]) -> set[int]:
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

def fetch_all_movies() -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tmdb_id, title, embedding, genres, keywords, 
                       director, top_cast, release_year
                FROM movies
                WHERE embedding IS NOT NULL
            """)
            return cur.fetchall()
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

def build_user_model(user_data: dict) -> Optional[dict]:
    user_id = user_data["id"]
    taste_vector = user_data["taste_vector"]
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
        enhanced_score = content_score + 0.3 * metadata_boost
        enhanced_scores.append(enhanced_score)
        ratings.append(movie["rating"])
    
    scores = np.array(enhanced_scores)
    ratings = np.array(ratings)
    
    return {
        "username": user_data["username"],
        "taste_vector": taste_vector,
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "min_rating": np.min(ratings),
        "max_rating": np.max(ratings),
        "keyword_scores": keyword_scores,
        "director_scores": director_scores,
        "cast_scores": cast_scores,
        "user_mean": user_mean
    }

def predict_for_user(movie: dict, user_model: dict) -> float:
    embedding = parse_embedding(movie["embedding"])
    
    content_score = calculate_content_score(user_model["taste_vector"], embedding)
    
    metadata_boost = calculate_metadata_boost(
        movie,
        user_model["keyword_scores"],
        user_model["director_scores"],
        user_model["cast_scores"],
        user_model["user_mean"]
    )
    
    enhanced_score = content_score + 0.3 * metadata_boost
    
    min_score = user_model["min_score"]
    max_score = user_model["max_score"]
    min_rating = user_model["min_rating"]
    max_rating = user_model["max_rating"]
    
    if max_score - min_score < 1e-6:
        return (min_rating + max_rating) / 2
    
    normalized = (enhanced_score - min_score) / (max_score - min_score)
    predicted = min_rating + normalized * (max_rating - min_rating)
    
    if enhanced_score < min_score:
        predicted = min_rating - (min_score - enhanced_score) * (max_rating - min_rating) / (max_score - min_score)
    elif enhanced_score > max_score:
        predicted = max_rating + (enhanced_score - max_score) * (max_rating - min_rating) / (max_score - min_score)
    
    return np.clip(predicted, 0.5, 5.0)

def recommend_for_group(
    usernames: list[str],
    top_n: int = 10,
    aggregation: str = "average"
) -> list[dict]:
    print(f"\n{'='*100}")
    print(f"GROUP RECOMMENDATIONS: {', '.join(usernames)}")
    print(f"{'='*100}\n")
    
    user_models = []
    for username in usernames:
        user_data = fetch_user_data(username)
        if not user_data:
            print(f"⚠️  User '{username}' not found or has no taste vector")
            continue
        
        model = build_user_model(user_data)
        if model:
            user_models.append(model)
            print(f"✓ Loaded model for {username}")
        else:
            print(f"⚠️  User '{username}' has insufficient ratings")
    
    if not user_models:
        print("\n❌ No valid users in group")
        return []
    
    print(f"\n📊 Using {len(user_models)} users for recommendations\n")
    
    watched_ids = fetch_group_watched_movies(usernames)
    all_movies = fetch_all_movies()
    
    predictions = []
    for movie in all_movies:
        if movie["tmdb_id"] in watched_ids:
            continue
        
        user_predictions = []
        for model in user_models:
            pred = predict_for_user(movie, model)
            user_predictions.append(pred)
        
        if aggregation == "average":
            group_rating = np.mean(user_predictions)
        elif aggregation == "min":
            group_rating = np.min(user_predictions)
        elif aggregation == "harmonic":
            group_rating = len(user_predictions) / sum(1/max(p, 0.5) for p in user_predictions)
        else:
            group_rating = np.mean(user_predictions)
        
        predictions.append({
            "tmdb_id": movie["tmdb_id"],
            "title": movie["title"],
            "year": movie.get("release_year"),
            "genres": movie["genres"] or [],
            "group_rating": group_rating,
            "individual_ratings": {
                user_models[i]["username"]: user_predictions[i] 
                for i in range(len(user_models))
            },
            "min_rating": min(user_predictions),
            "max_rating": max(user_predictions),
            "rating_variance": np.var(user_predictions)
        })
    
    predictions.sort(key=lambda x: x["group_rating"], reverse=True)
    
    return predictions[:top_n]

def print_group_recommendations(recommendations: list[dict]):
    print(f"\n🎬 TOP GROUP PICKS")
    print(f"{'='*100}\n")
    
    for i, rec in enumerate(recommendations, 1):
        year = f"({rec['year']})" if rec['year'] else ""
        genres = ", ".join(rec["genres"][:2])
        
        print(f"{i:2d}. {rec['title']} {year} - {rec['group_rating']:.1f}★")
        print(f"    {genres}")
        
        ratings_str = " | ".join([
            f"{user}: {rating:.1f}★" 
            for user, rating in rec["individual_ratings"].items()
        ])
        print(f"    {ratings_str}")
        
        variance = rec["rating_variance"]
        if variance < 0.25:
            agreement = "🟢 Strong agreement"
        elif variance < 1.0:
            agreement = "🟡 Moderate agreement"
        else:
            agreement = "🔴 Mixed opinions"
        
        print(f"    {agreement} (variance: {variance:.2f})\n")

if __name__ == "__main__":
    print("\n" + "="*100)
    print("GROUP MOVIE RECOMMENDER")
    print("Uses metadata-enhanced predictions (0.455 avg MAE)")
    print("="*100)
    
    usernames_input = input("\nEnter usernames (comma-separated): ").strip()
    usernames = [u.strip() for u in usernames_input.split(",") if u.strip()]
    
    if len(usernames) < 2:
        print("Need at least 2 users for group recommendations!")
    else:
        aggregation = input("\nAggregation method (average/min/harmonic) [average]: ").strip() or "average"
        top_n = int(input("How many recommendations? [10]: ").strip() or "10")
        
        recommendations = recommend_for_group(usernames, top_n=top_n, aggregation=aggregation)
        
        if recommendations:
            print_group_recommendations(recommendations)
            
            print(f"{'='*100}")
            print(f"RECOMMENDATION STATS")
            print(f"{'='*100}")
            
            unwatched_count = len([
                m for m in fetch_all_movies() 
                if m['tmdb_id'] not in fetch_group_watched_movies(usernames)
            ])
            
            print(f"Total unwatched movies analyzed: {unwatched_count}")
            print(f"Average group rating: {np.mean([r['group_rating'] for r in recommendations]):.2f}★")
            print(f"Average individual variance: {np.mean([r['rating_variance'] for r in recommendations]):.3f}")
        else:
            print("\n❌ Could not generate recommendations")