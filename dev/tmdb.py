import requests
from typing import Optional

from dev.config import TMDB_API_KEY

BASE_URL = "https://api.themoviedb.org/3"

def fetch_tmdb_movie(tmdb_id: int, append_to_response: Optional[str] = None) -> Optional[dict]:
    params = {"api_key": TMDB_API_KEY}
    if append_to_response:
        params["append_to_response"] = append_to_response

    resp = requests.get(f"{BASE_URL}/movie/{tmdb_id}", params=params)
    
    if resp.status_code != 200:
        return None

    return resp.json()