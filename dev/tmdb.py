import requests
from typing import Optional
from dev.config import TMDB_API_KEY

TMDB_BASE_URL = "https://api.themoviedb.org/3"

def get_tmdb_movie(tmdb_id: int, append_to_response: Optional[str] = None) -> Optional[dict]:
    """
    Fetch a TMDB movie by ID, optionally appending extra data like credits, keywords, recommendations
    """
    params = {"api_key": TMDB_API_KEY}
    if append_to_response:
        params["append_to_response"] = append_to_response

    resp = requests.get(f"{TMDB_BASE_URL}/movie/{tmdb_id}", params=params)
    if resp.status_code != 200:
        print(f"TMDB fetch failed for {tmdb_id}: {resp.status_code}")
        return None

    return resp.json()