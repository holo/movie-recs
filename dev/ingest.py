import time
from typing import Optional

from dev.db import get_conn
from dev.letterboxd import fetch_letterboxd_movie
from dev.tmdb import fetch_tmdb_movie
from dev.embeddings import embed_movies, update_user_taste_vectors
from dev.config import LETTERBOXD_DELAY

def upsert_user(conn, username: str) -> int:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO users (username)
            VALUES (%s)
            ON CONFLICT (username) DO NOTHING
            RETURNING id
        """, (username,))
        
        row = cur.fetchone()
        if row:
            return row["id"]

        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        return cur.fetchone()["id"]

def upsert_letterboxd_map(
    conn,
    letterboxd_id: int,
    letterboxd_slug: str,
    tmdb_id: Optional[int] = None,
):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO letterboxd_tmdb_map (letterboxd_id, letterboxd_slug, tmdb_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (letterboxd_id) DO UPDATE SET
                letterboxd_slug = EXCLUDED.letterboxd_slug,
                tmdb_id = COALESCE(EXCLUDED.tmdb_id, letterboxd_tmdb_map.tmdb_id),
                last_checked = NOW()
        """, (letterboxd_id, letterboxd_slug, tmdb_id))

def upsert_letterboxd_movie(
    conn,
    tmdb_id: int,
    letterboxd_id: int,
    letterboxd_slug: str,
    imdb_id: Optional[str] = None,
    avg_rating: Optional[float] = None,
    rating_count: Optional[int] = None,
    themes: Optional[list[str]] = None,
):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO letterboxd_movies
            (tmdb_id, letterboxd_id, letterboxd_slug, imdb_id, avg_rating, rating_count, themes, last_scraped)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (tmdb_id) DO UPDATE SET
                letterboxd_id = EXCLUDED.letterboxd_id,
                letterboxd_slug = EXCLUDED.letterboxd_slug,
                imdb_id = COALESCE(EXCLUDED.imdb_id, letterboxd_movies.imdb_id),
                avg_rating = COALESCE(EXCLUDED.avg_rating, letterboxd_movies.avg_rating),
                rating_count = COALESCE(EXCLUDED.rating_count, letterboxd_movies.rating_count),
                themes = COALESCE(EXCLUDED.themes, letterboxd_movies.themes),
                last_scraped = NOW()
        """, (tmdb_id, letterboxd_id, letterboxd_slug, imdb_id, avg_rating, rating_count, themes))

def upsert_user_interaction(
    conn,
    user_id: int,
    tmdb_id: int,
    rating: Optional[float] = None,
    liked: Optional[bool] = None,
    source: str = "letterboxd",
):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO user_movie_interactions
            (user_id, tmdb_id, rating, liked, source, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id, tmdb_id) DO UPDATE SET
                rating = EXCLUDED.rating,
                liked = EXCLUDED.liked,
                source = EXCLUDED.source,
                created_at = NOW()
        """, (user_id, tmdb_id, rating, liked, source))

def parse_release_year(release_date: Optional[str]) -> Optional[int]:
    if not release_date:
        return None
    return int(release_date[:4])

def extract_director(credits: dict) -> Optional[str]:
    for crew_member in credits.get("crew", []):
        if crew_member.get("job") == "Director":
            return crew_member.get("name")
    return None

def extract_top_cast(credits: dict, limit: int = 5) -> list[str]:
    return [c.get("name") for c in credits.get("cast", [])[:limit]]

def upsert_movie(conn, tmdb_data: dict):
    genres = [g["name"] for g in tmdb_data.get("genres", [])]
    keywords = [k["name"] for k in tmdb_data.get("keywords", {}).get("keywords", [])]
    director = extract_director(tmdb_data.get("credits", {}))
    top_cast = extract_top_cast(tmdb_data.get("credits", {}))

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO movies
            (tmdb_id, title, release_year, runtime, genres, director, top_cast, keywords,
             overview, tmdb_vote_avg, tmdb_vote_count, tmdb_popularity,
             poster_path, backdrop_path, embedding, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NOW(), NOW())
            ON CONFLICT (tmdb_id) DO UPDATE SET
                title = EXCLUDED.title,
                release_year = EXCLUDED.release_year,
                runtime = EXCLUDED.runtime,
                genres = EXCLUDED.genres,
                director = EXCLUDED.director,
                top_cast = EXCLUDED.top_cast,
                keywords = EXCLUDED.keywords,
                overview = EXCLUDED.overview,
                tmdb_vote_avg = EXCLUDED.tmdb_vote_avg,
                tmdb_vote_count = EXCLUDED.tmdb_vote_count,
                tmdb_popularity = EXCLUDED.tmdb_popularity,
                poster_path = EXCLUDED.poster_path,
                backdrop_path = EXCLUDED.backdrop_path,
                updated_at = NOW()
        """, (
            tmdb_data["id"],
            tmdb_data.get("title"),
            parse_release_year(tmdb_data.get("release_date")),
            tmdb_data.get("runtime"),
            genres,
            director,
            top_cast,
            keywords,
            tmdb_data.get("overview"),
            tmdb_data.get("vote_average"),
            tmdb_data.get("vote_count"),
            tmdb_data.get("popularity"),
            tmdb_data.get("poster_path"),
            tmdb_data.get("backdrop_path"),
        ))

def fetch_tmdb_id_from_map(conn, letterboxd_id: int) -> Optional[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT tmdb_id FROM letterboxd_tmdb_map WHERE letterboxd_id = %s", (letterboxd_id,))
        row = cur.fetchone()
        return row["tmdb_id"] if row else None

def movie_exists(conn, tmdb_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM movies WHERE tmdb_id = %s", (tmdb_id,))
        return cur.fetchone() is not None

def fetch_tmdb_id_for_rating(conn, rating: dict) -> Optional[int]:
    tmdb_id = fetch_tmdb_id_from_map(conn, rating["letterboxd_id"])
    
    if not tmdb_id:
        movie_data = fetch_letterboxd_movie(rating["letterboxd_slug"])
        if movie_data:
            tmdb_id = movie_data["tmdb_id"]
            rating.update(movie_data)
        time.sleep(LETTERBOXD_DELAY)
    
    return tmdb_id

def fetch_and_insert_tmdb_movie(conn, tmdb_id: int) -> Optional[list[str]]:
    if movie_exists(conn, tmdb_id):
        return []

    tmdb_data = fetch_tmdb_movie(tmdb_id, append_to_response="credits,keywords,recommendations")
    if not tmdb_data:
        return None

    upsert_movie(conn, tmdb_data)
    
    recs = tmdb_data.get("recommendations", {}).get("results", [])
    return [m["title"] for m in recs]

def ingest_letterboxd_user(username: str, ratings: list[dict]) -> list[str]:
    conn = get_conn()
    recommendations = []

    try:
        user_id = upsert_user(conn, username)

        for rating in ratings:
            tmdb_id = fetch_tmdb_id_for_rating(conn, rating)

            if not tmdb_id:
                print(f"Skipping {rating['letterboxd_slug']} — could not determine TMDB ID")
                continue

            recs = fetch_and_insert_tmdb_movie(conn, tmdb_id)
            if recs is None:
                print(f"TMDB data not found for TMDB ID {tmdb_id}, skipping movie {rating['letterboxd_slug']}")
                continue

            recommendations.extend(recs)

            upsert_letterboxd_map(conn, rating["letterboxd_id"], rating["letterboxd_slug"], tmdb_id)

            upsert_letterboxd_movie(
                conn,
                tmdb_id=tmdb_id,
                letterboxd_id=rating["letterboxd_id"],
                letterboxd_slug=rating["letterboxd_slug"],
                imdb_id=rating.get("imdb_id"),
                avg_rating=rating.get("avg_rating"),
                rating_count=rating.get("rating_count"),
                themes=rating.get("themes"),
            )

            upsert_user_interaction(
                conn,
                user_id=user_id,
                tmdb_id=tmdb_id,
                rating=rating.get("rating"),
                liked=rating.get("liked"),
                source=rating.get("source", "letterboxd"),
            )

        conn.commit()
    finally:
        conn.close()

    embed_movies()
    update_user_taste_vectors()

    return recommendations