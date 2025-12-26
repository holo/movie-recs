import time
import requests
from bs4 import BeautifulSoup
from typing import Optional

from dev.config import USER_AGENT, LETTERBOXD_DELAY

BASE_URL = "https://letterboxd.com"

def fetch_user_ratings(username: str) -> list[dict]:
    results = []
    page = 1

    headers = {"User-Agent": USER_AGENT}

    while True:
        url = f"{BASE_URL}/{username}/films/page/{page}/"
        resp = requests.get(url, headers=headers)

        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("li.griditem")

        if not items:
            break

        for item in items:
            parsed = parse_grid_item(item)
            if parsed:
                results.append(parsed)

        page += 1
        time.sleep(LETTERBOXD_DELAY)

    return results

def parse_grid_item(item) -> Optional[dict]:
    component = item.select_one(".react-component")
    if not component:
        return None

    slug = component.get("data-item-slug")
    film_id = component.get("data-film-id")

    if not slug:
        return None

    rating = parse_star_rating(item)
    liked = bool(item.select_one(".icon-like"))

    return {
        "letterboxd_slug": slug,
        "letterboxd_film_id": int(film_id) if film_id else None,
        "rating": rating,
        "liked": liked,
    }

def parse_star_rating(item) -> Optional[float]:
    rating_span = item.select_one("span.rating")
    if not rating_span:
        return None

    text = rating_span.get_text(strip=True)

    full_stars = text.count("★")
    half_star = "½" in text

    return full_stars + (0.5 if half_star else 0.0)
