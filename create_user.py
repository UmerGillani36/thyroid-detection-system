import sqlite3

# Connect or create the DB
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
""")

# Insert a user manually
cursor.execute("INSERT OR REPLACE INTO users (username, password) VALUES (?, ?)", ("admin", "1234"))

conn.commit()
conn.close()

print("User created.")
