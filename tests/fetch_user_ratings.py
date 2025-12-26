from dev.letterboxd import fetch_user_ratings

username = "alfie"

ratings = fetch_user_ratings(username)

print(f"fetched {len(ratings)} ratings\n")

for movie in ratings[:10]:
    print(movie)
