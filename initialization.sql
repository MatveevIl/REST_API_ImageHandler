CREATE TABLE IF NOT EXISTS resize (
    id SERIAL PRIMARY KEY,
    source BYTEA,
    source_path TEXT,
    result BYTEA,
    result_path TEXT
);

CREATE TABLE IF NOT EXISTS overlay (
    id SERIAL PRIMARY KEY,
    source BYTEA,
    source_path TEXT,
    image2 BYTEA,
    image2_path TEXT,
    merge_value INTEGER,
    result BYTEA,
    result_path TEXT
);

CREATE TABLE IF NOT EXISTS face_finding (
    id SERIAL PRIMARY KEY,
    source BYTEA,
    source_path TEXT,
    faces INTEGER,
    result BYTEA,
    result_path TEXT
);

