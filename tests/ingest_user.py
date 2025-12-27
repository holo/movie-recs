from dev.letterboxd import fetch_user_ratings
from dev.ingest import ingest_letterboxd_user

# username = "alfie"
username = input('username: ')

ratings = fetch_user_ratings(username)
print(f"fetched {len(ratings)} ratings for {username}\n")

for r in ratings[:5]:
    print(r)

recommendations = ingest_letterboxd_user(username, ratings)
print(f"\ncollected {len(recommendations)} tmdb recommendations for future scraping")
print("sample tmdb recommendations:", recommendations[:10])
