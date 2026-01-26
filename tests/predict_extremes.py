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


def fetch_user_data(username: str) -> Optional[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, taste_vector FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            
            if not row or row["taste_vector"] is None:
                return None
            
            return {
                "id": row["id"],
                "taste_vector": np.array(ast.literal_eval(row["taste_vector"]), dtype=np.float32)
            }
    finally:
        conn.close()


def fetch_all_movies_with_metadata() -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tmdb_id, title, embedding, genres, keywords, director, top_cast, release_year
                FROM movies
                WHERE embedding IS NOT NULL
            """)
            return cur.fetchall()
    finally:
        conn.close()


def fetch_watched_movies(user_id: int) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT umi.tmdb_id, m.title, m.embedding, umi.rating, 
                       m.genres, m.keywords, m.director, m.top_cast, m.release_year
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
        keywords = movie["keywords"] or []
        
        for kw in keywords:
            if kw not in keyword_ratings:
                keyword_ratings[kw] = []
            keyword_ratings[kw].append(rating)
    
    keyword_scores = {}
    for kw, ratings in keyword_ratings.items():
        if len(ratings) >= 2:
            avg_rating = np.mean(ratings)
            keyword_scores[kw] = avg_rating
    
    return keyword_scores


def learn_director_preferences(watched_movies: list[dict]) -> dict:
    director_ratings = {}
    
    for movie in watched_movies:
        rating = movie["rating"]
        director = movie["director"]
        
        if director:
            if director not in director_ratings:
                director_ratings[director] = []
            director_ratings[director].append(rating)
    
    director_scores = {}
    for director, ratings in director_ratings.items():
        if len(ratings) >= 1:
            avg_rating = np.mean(ratings)
            director_scores[director] = avg_rating
    
    return director_scores


def learn_cast_preferences(watched_movies: list[dict]) -> dict:
    cast_ratings = {}
    
    for movie in watched_movies:
        rating = movie["rating"]
        cast = movie["top_cast"] or []
        
        for actor in cast:
            if actor not in cast_ratings:
                cast_ratings[actor] = []
            cast_ratings[actor].append(rating)
    
    cast_scores = {}
    for actor, ratings in cast_ratings.items():
        if len(ratings) >= 2:
            avg_rating = np.mean(ratings)
            cast_scores[actor] = avg_rating
    
    return cast_scores


def calculate_metadata_boost_with_explanation(
    movie: dict,
    keyword_scores: dict,
    director_scores: dict,
    cast_scores: dict,
    user_mean: float
) -> Tuple[float, dict]:
    boost = 0.0
    feature_count = 0
    
    explanation = {
        "positive_signals": [],
        "negative_signals": [],
        "director_signal": None,
        "cast_signals": []
    }
    
    keywords = movie.get("keywords") or []
    if keywords:
        keyword_boosts = []
        for kw in keywords:
            if kw in keyword_scores:
                kw_score = keyword_scores[kw]
                delta = kw_score - user_mean
                keyword_boosts.append(delta)
                
                if delta > 0.3:
                    explanation["positive_signals"].append(f"{kw} (you rate this {kw_score:.1f}★)")
                elif delta < -0.3:
                    explanation["negative_signals"].append(f"{kw} (you rate this {kw_score:.1f}★)")
        
        if keyword_boosts:
            boost += np.mean(keyword_boosts)
            feature_count += 1
    
    director = movie.get("director")
    if director and director in director_scores:
        dir_score = director_scores[director]
        delta = dir_score - user_mean
        boost += delta
        feature_count += 1
        
        if abs(delta) > 0.2:
            explanation["director_signal"] = f"{director} (avg {dir_score:.1f}★ from you)"
    
    cast = movie.get("top_cast") or []
    if cast:
        cast_boosts = []
        for actor in cast:
            if actor in cast_scores:
                actor_score = cast_scores[actor]
                delta = actor_score - user_mean
                cast_boosts.append(delta)
                
                if abs(delta) > 0.3:
                    explanation["cast_signals"].append(f"{actor} (avg {actor_score:.1f}★ from you)")
        
        if cast_boosts:
            boost += np.mean(cast_boosts)
            feature_count += 1
    
    if feature_count > 0:
        return boost / feature_count, explanation
    
    return 0.0, explanation


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
        metadata_boost, _ = calculate_metadata_boost_with_explanation(
            movie, keyword_scores, director_scores, cast_scores, user_mean
        )
        enhanced_score = content_score + 0.3 * metadata_boost
        
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


def predict_rating_enhanced(
    content_score: float,
    movie: dict,
    mapping_info: Optional[dict]
) -> Tuple[float, dict]:
    explanation = {
        "content_similarity": content_score,
        "metadata_boost": 0.0,
        "positive_signals": [],
        "negative_signals": [],
        "director_signal": None,
        "cast_signals": []
    }
    
    if mapping_info is None:
        predicted = 2.75 + content_score * 2.25
        return np.clip(predicted, 0.5, 5.0), explanation
    
    metadata_boost, meta_explanation = calculate_metadata_boost_with_explanation(
        movie,
        mapping_info["keyword_scores"],
        mapping_info["director_scores"],
        mapping_info["cast_scores"],
        mapping_info["user_mean"]
    )
    
    explanation["metadata_boost"] = metadata_boost
    explanation["positive_signals"] = meta_explanation["positive_signals"]
    explanation["negative_signals"] = meta_explanation["negative_signals"]
    explanation["director_signal"] = meta_explanation["director_signal"]
    explanation["cast_signals"] = meta_explanation["cast_signals"]
    
    enhanced_score = content_score + 0.3 * metadata_boost
    
    min_score = mapping_info["min_score"]
    max_score = mapping_info["max_score"]
    min_rating = mapping_info["min_rating"]
    max_rating = mapping_info["max_rating"]
    
    if max_score - min_score < 1e-6:
        predicted = (min_rating + max_rating) / 2
    else:
        normalized = (enhanced_score - min_score) / (max_score - min_score)
        predicted = min_rating + normalized * (max_rating - min_rating)
        
        if enhanced_score < min_score:
            predicted = min_rating - (min_score - enhanced_score) * (max_rating - min_rating) / (max_score - min_score)
        elif enhanced_score > max_score:
            predicted = max_rating + (enhanced_score - max_score) * (max_rating - min_rating) / (max_score - min_score)
    
    return np.clip(predicted, 0.5, 5.0), explanation


def format_explanation(explanation: dict, show_all: bool = False) -> str:
    parts = []
    
    if explanation["positive_signals"]:
        top_positive = explanation["positive_signals"][:3] if not show_all else explanation["positive_signals"]
        parts.append("+ " + ", ".join(top_positive))
    
    if explanation["director_signal"]:
        parts.append(f"+ Director: {explanation['director_signal']}")
    
    if explanation["cast_signals"]:
        top_cast = explanation["cast_signals"][:2] if not show_all else explanation["cast_signals"]
        if top_cast:
            parts.append("+ Cast: " + ", ".join(top_cast))
    
    if explanation["negative_signals"]:
        top_negative = explanation["negative_signals"][:3] if not show_all else explanation["negative_signals"]
        parts.append("- " + ", ".join(top_negative))
    
    if not parts:
        return "Based on content similarity"
    
    return " | ".join(parts)


def predict_unwatched_extremes(username: str, top_n: int = 10):
    user_data = fetch_user_data(username)
    if not user_data:
        print(f"User '{username}' not found or has no taste vector.")
        return

    taste_vector = user_data["taste_vector"]
    user_id = user_data["id"]
    
    mapping_info = fit_enhanced_mapping(user_id, taste_vector)
    
    if not mapping_info:
        print("Insufficient data to make predictions.")
        return
    
    print(f"\n{'='*100}")
    print(f"MOVIE RECOMMENDATIONS FOR: {username}")
    print(f"{'='*100}")
    
    watched_ids = {m["tmdb_id"] for m in fetch_watched_movies(user_id)}
    all_movies = fetch_all_movies_with_metadata()

    predictions = []
    for movie in all_movies:
        if movie["tmdb_id"] in watched_ids:
            continue
        
        embedding = parse_embedding(movie["embedding"])
        content_score = calculate_content_score(taste_vector, embedding)
        predicted_rating, explanation = predict_rating_enhanced(content_score, movie, mapping_info)
        
        predictions.append({
            "tmdb_id": movie["tmdb_id"],
            "title": movie["title"],
            "year": movie.get("release_year"),
            "predicted_rating": predicted_rating,
            "genres": movie["genres"] or [],
            "explanation": explanation
        })

    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    
    highest = predictions[:top_n]
    lowest = predictions[-top_n:][::-1]

    print(f"\n🎬 TOP {top_n} RECOMMENDED (You'll Love These)")
    print(f"{'='*100}\n")
    
    for i, movie in enumerate(highest, 1):
        year = f"({movie['year']})" if movie['year'] else ""
        genres = ", ".join(movie["genres"][:2])
        
        print(f"{i:2d}. {movie['title']} {year} - {movie['predicted_rating']:.1f}★")
        print(f"    {genres}")
        print(f"    {format_explanation(movie['explanation'])}\n")

    print(f"\n🚫 BOTTOM {top_n} (Skip These)")
    print(f"{'='*100}\n")
    
    for i, movie in enumerate(lowest, 1):
        year = f"({movie['year']})" if movie['year'] else ""
        genres = ", ".join(movie["genres"][:2])
        
        print(f"{i:2d}. {movie['title']} {year} - {movie['predicted_rating']:.1f}★")
        print(f"    {genres}")
        print(f"    {format_explanation(movie['explanation'])}\n")


def predict_watched_extremes(username: str, top_n: int = 10):
    user_data = fetch_user_data(username)
    if not user_data:
        print(f"User '{username}' not found or has no taste vector.")
        return

    taste_vector = user_data["taste_vector"]
    user_id = user_data["id"]
    
    mapping_info = fit_enhanced_mapping(user_id, taste_vector)
    watched_movies = fetch_watched_movies(user_id)

    predictions = []
    errors = []
    
    for movie in watched_movies:
        embedding = parse_embedding(movie["embedding"])
        content_score = calculate_content_score(taste_vector, embedding)
        predicted_rating, explanation = predict_rating_enhanced(content_score, movie, mapping_info)
        
        predictions.append({
            "tmdb_id": movie["tmdb_id"],
            "title": movie["title"],
            "year": movie.get("release_year"),
            "predicted_rating": predicted_rating,
            "actual_rating": movie["rating"],
            "genres": movie["genres"] or [],
            "explanation": explanation
        })
        
        error = abs(predicted_rating - movie["rating"])
        errors.append(error)

    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    
    highest = predictions[:top_n]
    lowest = predictions[-top_n:][::-1]

    print(f"\n{'='*100}")
    print(f"PREDICTION ACCURACY CHECK - HIGHEST PREDICTED")
    print(f"{'='*100}\n")
    
    for i, movie in enumerate(highest, 1):
        diff = movie['predicted_rating'] - movie['actual_rating']
        year = f"({movie['year']})" if movie['year'] else ""
        accuracy = "+" if abs(diff) < 0.5 else "-"
        
        print(f"{i:2d}. {movie['title']} {year}")
        print(f"    Predicted: {movie['predicted_rating']:.1f}★ | Actual: {movie['actual_rating']:.1f}★ | Error: {diff:+.1f} {accuracy}")
        print(f"    {format_explanation(movie['explanation'])}\n")

    print(f"{'='*100}")
    print(f"PREDICTION ACCURACY CHECK - LOWEST PREDICTED")
    print(f"{'='*100}\n")
    
    for i, movie in enumerate(lowest, 1):
        diff = movie['predicted_rating'] - movie['actual_rating']
        year = f"({movie['year']})" if movie['year'] else ""
        accuracy = "+" if abs(diff) < 0.5 else "-"
        
        print(f"{i:2d}. {movie['title']} {year}")
        print(f"    Predicted: {movie['predicted_rating']:.1f}★ | Actual: {movie['actual_rating']:.1f}★ | Error: {diff:+.1f} {accuracy}")
        print(f"    {format_explanation(movie['explanation'])}\n")
    
    if errors:
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        
        print(f"{'='*100}")
        print(f"OVERALL ACCURACY")
        print(f"{'='*100}")
        print(f"Mean Absolute Error: {mae:.3f}★")
        print(f"Root Mean Squared Error: {rmse:.3f}★")
        print(f"Predictions within 0.5★: {sum(1 for e in errors if e < 0.5) / len(errors) * 100:.1f}%")
        print(f"Predictions within 1.0★: {sum(1 for e in errors if e < 1.0) / len(errors) * 100:.1f}%")


if __name__ == "__main__":
    username = input("Username: ").strip()
    
    predict_unwatched_extremes(username)
    print("\n")
    predict_watched_extremes(username)