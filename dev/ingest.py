import time
from typing import Optional, List
from dev.db import get_conn
from dev.letterboxd import fetch_letterboxd_movie
from dev.tmdb import get_tmdb_movie
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

def insert_movie(conn, tmdb_data: dict):
    """
    Map TMDB response to your movies table and insert
    """
    with conn.cursor() as cur:
        genres = [g["name"] for g in tmdb_data.get("genres", [])]
        director = None
        top_cast = []
        keywords = [k["name"] for k in tmdb_data.get("keywords", {}).get("keywords", [])]

        # get director from credits
        for c in tmdb_data.get("credits", {}).get("crew", []):
            if c.get("job") == "Director":
                director = c.get("name")
                break

        # top 5 cast
        top_cast = [c.get("name") for c in tmdb_data.get("credits", {}).get("cast", [])[:5]]

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
            int(tmdb_data.get("release_date", "0000-00-00")[:4]) if tmdb_data.get("release_date") else None,
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

# ----------------------------
# Full ingestion function
# ----------------------------
def ingest_letterboxd_user(username: str, ratings: List[dict]) -> List[int]:
    """
    Ingest a Letterboxd user fully:
      - ensures user exists
      - fetches missing Letterboxd movie data
      - fetches TMDB data if missing
      - inserts into movies, letterboxd_movies, user interactions
      - returns TMDB recommendations (list of IDs) for potential future ingestion
    """
    conn = get_conn()
    recommendations = []

    try:
        user_id = upsert_user(conn, username)

        for r in ratings:
            # check if tmdb_id exists in map
            tmdb_id = None
            with conn.cursor() as cur:
                cur.execute("SELECT tmdb_id FROM letterboxd_tmdb_map WHERE letterboxd_id = %s", (r["letterboxd_id"],))
                row = cur.fetchone()
                if row:
                    tmdb_id = row["tmdb_id"]

            # fetch letterboxd movie data if tmdb_id is missing
            if not tmdb_id:
                movie_data = fetch_letterboxd_movie(r["letterboxd_slug"])
                if movie_data:
                    tmdb_id = movie_data["tmdb_id"]
                    r.update(movie_data)
                time.sleep(LETTERBOXD_DELAY)

            if not tmdb_id:
                print(f"Skipping {r['letterboxd_slug']} — could not determine TMDB ID")
                continue

            # fetch TMDB data if movie doesn't exist yet
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM movies WHERE tmdb_id = %s", (tmdb_id,))
                if not cur.fetchone():
                    tmdb_data = get_tmdb_movie(tmdb_id, append_to_response="credits,keywords,recommendations")
                    if tmdb_data:
                        insert_movie(conn, tmdb_data)
                        # collect recommendation IDs
                        recs = tmdb_data.get("recommendations", {}).get("results", [])
                        recommendations.extend([m["title"] for m in recs])

            # upsert letterboxd map
            upsert_letterboxd_map(conn, r["letterboxd_id"], r["letterboxd_slug"], tmdb_id)

            # upsert letterboxd_movies
            upsert_letterboxd_movie(
                conn,
                tmdb_id=tmdb_id,
                letterboxd_id=r["letterboxd_id"],
                letterboxd_slug=r["letterboxd_slug"],
                imdb_id=r.get("imdb_id"),
                avg_rating=r.get("avg_rating"),
                rating_count=r.get("rating_count"),
                themes=r.get("themes"),
            )

            # upsert user interaction
            upsert_user_interaction(
                conn,
                user_id=user_id,
                tmdb_id=tmdb_id,
                rating=r.get("rating"),
                liked=r.get("liked"),
                source=r.get("source", "letterboxd"),
            )

        conn.commit()
    finally:
        conn.close()

    return recommendations