import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

USER_AGENT = "MovieRecs/0.1 (personal project)"
LETTERBOXD_DELAY = 1.5