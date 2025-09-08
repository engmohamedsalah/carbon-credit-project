#!/usr/bin/env python3
"""
Update user passwords to match USERS.md documentation
"""

import sqlite3
from passlib.context import CryptContext

# Initialize password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)

def update_user_password(email: str, new_password: str):
    """Update a user's password in the database."""
    db_path = "../database/carbon_credits.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Hash the new password
            hashed_password = hash_password(new_password)
            
            # Update the user's password
            cursor = conn.execute(
                "UPDATE users SET hashed_password = ? WHERE email = ?",
                (hashed_password, email)
            )
            
            if cursor.rowcount > 0:
                print(f"✅ Updated password for {email}")
            else:
                print(f"❌ User {email} not found in database")
                
            conn.commit()
            
    except Exception as e:
        print(f"❌ Error updating password for {email}: {e}")

def main():
    """Update passwords for key users from USERS.md"""
    print("🔐 Updating user passwords to match USERS.md documentation...")
    
    # Key users from USERS.md with their documented passwords
    users_to_update = [
        ("admin@admin.com", "admin123"),
        ("blockchaintest@example.com", "password123"),
    ]
    
    for email, password in users_to_update:
        update_user_password(email, password)
    
    print("✅ Password update process completed!")
    print("\nUpdated credentials:")
    for email, password in users_to_update:
        print(f"  - {email} / {password}")

if __name__ == "__main__":
    main()