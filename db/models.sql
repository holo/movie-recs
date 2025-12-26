CREATE EXTENSION IF NOT EXISTS vector;

-- movies
CREATE TABLE movies (
    tmdb_id INTEGER PRIMARY KEY,

    title TEXT NOT NULL,
    release_year INTEGER NOT NULL,
    runtime INTEGER,

    genres TEXT[],
    director TEXT,
    top_cast TEXT[],
    keywords TEXT[],

    overview TEXT,

    tmdb_vote_avg REAL,
    tmdb_vote_count INTEGER,
    tmdb_popularity REAL,

    poster_path TEXT,
    backdrop_path TEXT,

    embedding VECTOR(768),

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- letterboxd slug/id <-> tmdb id
CREATE TABLE letterboxd_tmdb_map (
    letterboxd_id INTEGER PRIMARY KEY,
    letterboxd_slug TEXT UNIQUE NOT NULL,

    tmdb_id INTEGER UNIQUE
        REFERENCES movies(tmdb_id)
        ON DELETE SET NULL,

    last_checked TIMESTAMP DEFAULT NOW()
);

-- letterboxd movie info
CREATE TABLE letterboxd_movies (
    tmdb_id INTEGER PRIMARY KEY
        REFERENCES movies(tmdb_id)
        ON DELETE CASCADE,

    letterboxd_id INTEGER UNIQUE NOT NULL
        REFERENCES letterboxd_tmdb_map(letterboxd_id)
        ON DELETE CASCADE,

    letterboxd_slug TEXT UNIQUE NOT NULL,

    imdb_id TEXT,
    avg_rating REAL,
    rating_count INTEGER,

    themes TEXT[],

    last_scraped TIMESTAMP DEFAULT NOW()
);

-- users
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,

    taste_vector VECTOR(768),

    created_at TIMESTAMP DEFAULT NOW()
);

-- user interactions
CREATE TABLE user_movie_interactions (
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    tmdb_id INTEGER REFERENCES movies(tmdb_id) ON DELETE CASCADE,

    rating REAL CHECK (rating >= 0 AND rating <= 5),
    liked BOOLEAN,

    source TEXT NOT NULL DEFAULT 'letterboxd',

    created_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (user_id, tmdb_id),

    CHECK (rating IS NOT NULL OR liked IS NOT NULL)
);

-- indexes
CREATE INDEX movies_embedding_idx
ON movies USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX user_movie_tmdb_idx
ON user_movie_interactions (tmdb_id);

CREATE INDEX user_movie_user_idx
ON user_movie_interactions (user_id);

CREATE INDEX letterboxd_map_tmdb_idx
ON letterboxd_tmdb_map (tmdb_id);