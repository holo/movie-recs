from dev.letterboxd import fetch_letterboxd_films, fetch_letterboxd_movie
from dev.ingest import upsert_letterboxd_map, upsert_letterboxd_movie, upsert_movie
from dev.tmdb import fetch_tmdb_movie
from dev.embeddings import embed_movies, update_user_taste_vectors
from dev.db import get_conn

SORT = "popular"
PAGES = 5

def film_already_mapped(conn, slug: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tmdb_id FROM letterboxd_tmdb_map WHERE letterboxd_slug = %s",
            (slug,)
        )
        row = cur.fetchone()
        return row and row["tmdb_id"] is not None

def process_film(conn, film: dict, seen_slugs: set[str]) -> bool:
    slug = film["letterboxd_slug"]

    if slug in seen_slugs:
        return False
    seen_slugs.add(slug)

    if film_already_mapped(conn, slug):
        return False

    lb_data = fetch_letterboxd_movie(slug)
    if not lb_data:
        return False

    tmdb_id = lb_data["tmdb_id"]

    tmdb_data = fetch_tmdb_movie(
        tmdb_id,
        append_to_response="credits,keywords,recommendations"
    )
    if not tmdb_data:
        return False

    upsert_movie(conn, tmdb_data)

    upsert_letterboxd_map(
        conn,
        lb_data["letterboxd_id"],
        lb_data["letterboxd_slug"],
        tmdb_id
    )

    upsert_letterboxd_movie(
        conn,
        tmdb_id=tmdb_id,
        letterboxd_id=lb_data["letterboxd_id"],
        letterboxd_slug=lb_data["letterboxd_slug"],
        imdb_id=lb_data.get("imdb_id"),
        avg_rating=lb_data.get("avg_rating"),
        rating_count=lb_data.get("rating_count"),
        themes=lb_data.get("themes"),
    )

    conn.commit()
    print(f"inserted: {slug}")
    return True

def ingest_letterboxd_films():
    conn = get_conn()
    seen_slugs = set()

    try:
        for page in range(1, PAGES + 1):
            films = fetch_letterboxd_films(sort=SORT, page=page)
            print(f"page {page}: {len(films)} films")

            for film in films:
                process_film(conn, film, seen_slugs)

    finally:
        conn.close()

    embed_movies()
    update_user_taste_vectors()

if __name__ == "__main__":
    ingest_letterboxd_films()