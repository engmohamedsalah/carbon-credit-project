"""
Simplified database initialization script for SQLite.
"""
import os
import sqlite3

# Create database directory if it doesn't exist
os.makedirs('database', exist_ok=True)

# Create SQLite database
db_path = 'database/carbon_credits.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create users table
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    hashed_password TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE
)
''')

# Create projects table
cursor.execute('''
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    location_name TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
''')

# Create verifications table
cursor.execute('''
CREATE TABLE IF NOT EXISTS verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    carbon_impact REAL,
    ai_confidence REAL,
    human_verified BOOLEAN DEFAULT FALSE,
    blockchain_certified BOOLEAN DEFAULT FALSE,
    certificate_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects (id)
)
''')

# Create satellite_images table
cursor.execute('''
CREATE TABLE IF NOT EXISTS satellite_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    image_url TEXT NOT NULL,
    acquisition_date TIMESTAMP NOT NULL,
    cloud_coverage REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects (id)
)
''')

# Check if admin user exists
cursor.execute("SELECT * FROM users WHERE email = 'admin@example.com'")
admin = cursor.fetchone()

# Add admin user if not exists
if not admin:
    # In a real app, this would be a proper password hash
    hashed_password = "password123_hashed"
    cursor.execute(
        "INSERT INTO users (email, full_name, hashed_password, is_active, is_admin) VALUES (?, ?, ?, ?, ?)",
        ("admin@example.com", "Admin User", hashed_password, True, True)
    )
    print("Admin user created.")

# Commit changes and close connection
conn.commit()
conn.close()

print("SQLite database initialized successfully.")
print(f"Database path: {os.path.abspath(db_path)}")
print("You can now start the backend server.") 