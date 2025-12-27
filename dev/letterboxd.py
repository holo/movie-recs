import time
import json
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
        "letterboxd_id": int(film_id) if film_id else None,
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

def fetch_letterboxd_movie(slug: str) -> Optional[dict]:
    url = f"{BASE_URL}/film/{slug}/"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers)
    
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    tmdb_id = parse_tmdb_id(soup)
    if not tmdb_id:
        return None

    letterboxd_data = parse_letterboxd_ids(soup)
    if not letterboxd_data:
        return None

    imdb_id = parse_imdb_id(soup)
    rating_data = parse_aggregate_rating(soup)
    themes = parse_themes(soup)

    time.sleep(LETTERBOXD_DELAY)

    return {
        "letterboxd_id": letterboxd_data["id"],
        "letterboxd_slug": letterboxd_data["slug"],
        "tmdb_id": tmdb_id,
        "imdb_id": imdb_id,
        "avg_rating": rating_data.get("avg_rating"),
        "rating_count": rating_data.get("rating_count"),
        "themes": themes,
    }

def parse_tmdb_id(soup) -> Optional[int]:
    body = soup.find("body", class_="film")
    if not body:
        return None
    
    tmdb_id_attr = body.get("data-tmdb-id")
    return int(tmdb_id_attr) if tmdb_id_attr else None

def parse_letterboxd_ids(soup) -> Optional[dict]:
    poster_component = soup.select_one("section.poster-list div.react-component")
    if not poster_component:
        return None

    film_id = poster_component.get("data-film-id")
    slug = poster_component.get("data-item-slug")

    if not film_id or not slug:
        return None

    return {
        "id": int(film_id),
        "slug": slug,
    }

def parse_imdb_id(soup) -> Optional[str]:
    imdb_link = soup.select_one('p.text-link.text-footer a[href*="imdb.com/title/"]')
    if not imdb_link:
        return None

    href = imdb_link.get("href", "")
    if "title/" not in href:
        return None

    return href.split("title/")[1].split("/")[0]

def parse_aggregate_rating(soup) -> dict:
    ld_json_tag = soup.find("script", type="application/ld+json")
    if not ld_json_tag or not ld_json_tag.string:
        return {}

    raw = ld_json_tag.string.strip()
    if raw.startswith("/*"):
        raw = raw.split("*/", 1)[-1].strip()
    if raw.endswith("*/"):
        raw = raw.rsplit("/*", 1)[0].strip()

    try:
        data = json.loads(raw)
        if "aggregateRating" in data:
            return {
                "avg_rating": data["aggregateRating"].get("ratingValue"),
                "rating_count": data["aggregateRating"].get("ratingCount"),
            }
    except Exception:
        pass

    return {}

def parse_themes(soup) -> list[str]:
    themes = []
    themes_header = soup.select_one("h3:-soup-contains('Themes')")
    
    if not themes_header:
        return themes

    themes_div = themes_header.find_next_sibling("div", class_="text-sluglist")
    if not themes_div:
        return themes

    theme_links = themes_div.select("p a.text-slug")
    for link in theme_links:
        text = link.get_text(strip=True)
        if not text.lower().startswith("show all"):
            themes.append(text)

    return themes

def fetch_letterboxd_films(
    sort: str = "popular",
    page: int = 1,
    genre: Optional[str] = None,
) -> list[dict]:
    headers = {"User-Agent": USER_AGENT}

    if sort == "popular":
        if genre:
            url = f"{BASE_URL}/films/ajax/popular/genre/{genre}/page/{page}/"
        else:
            url = f"{BASE_URL}/films/ajax/popular/page/{page}/"
    else:
        raise ValueError(f"unsupported sort: {sort}")

    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    items = soup.select("li.posteritem")
    results = []

    for item in items:
        component = item.select_one(".react-component")
        if not component:
            continue

        slug = component.get("data-item-slug")
        film_id = component.get("data-film-id")

        if not slug:
            continue

        results.append({
            "letterboxd_slug": slug,
            "letterboxd_id": int(film_id) if film_id else None,
        })

    time.sleep(LETTERBOXD_DELAY)
    return results