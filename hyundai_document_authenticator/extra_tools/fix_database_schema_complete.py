#!/usr/bin/env python3
"""
Complete Database Schema Fix Script
===================================
This script checks the current database schema and adds any missing columns
to match the User model definition.
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('../.env')

def get_database_connection():
    """Get database connection from environment variables."""
    host = os.getenv('POSTGRES_USER_HOST', 'localhost')
    port = os.getenv('POSTGRES_USER_PORT', '5432')
    database = os.getenv('POSTGRES_USER_DB', 'image_similarity_users_db')
    user = os.getenv('POSTGRES_USER_USER', 'postgres')
    password = os.getenv('POSTGRES_USER_PASSWORD', '')
    
    print(f"Connecting to PostgreSQL:")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Database: {database}")
    print(f"  User: {user}")
    
    return psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )

def get_table_columns(cursor, table_name):
    """Get all columns in a table."""
    cursor.execute("""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns 
        WHERE table_name = %s
        ORDER BY ordinal_position
    """, (table_name,))
    return cursor.fetchall()

def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table."""
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = %s AND column_name = %s
    """, (table_name, column_name))
    return cursor.fetchone() is not None

def add_missing_columns():
    """Add any missing columns to the users table."""
    # Expected columns based on the User model
    expected_columns = {
        'id': 'SERIAL PRIMARY KEY',
        'username': 'VARCHAR(64) UNIQUE NOT NULL',
        'email': 'VARCHAR(120) UNIQUE NOT NULL',
        'password_hash': 'VARCHAR(256) NOT NULL',
        'role': 'VARCHAR(20) NOT NULL DEFAULT \'user\'',
        'email_validated': 'BOOLEAN NOT NULL DEFAULT FALSE',
        'created_at': 'TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP',
        'last_login_at': 'TIMESTAMP WITH TIME ZONE'
    }
    
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        print("\nChecking current table schema...")
        current_columns = get_table_columns(cursor, 'users')
        current_column_names = [col[0] for col in current_columns]
        
        print(f"Current columns in 'users' table:")
        for col in current_columns:
            print(f"  - {col[0]} ({col[1]}) {'NULL' if col[2] == 'YES' else 'NOT NULL'}")
        
        print(f"\nExpected columns:")
        for col_name, col_def in expected_columns.items():
            print(f"  - {col_name}")
        
        # Find missing columns
        missing_columns = []
        for col_name in expected_columns.keys():
            if col_name not in current_column_names:
                missing_columns.append(col_name)
        
        if not missing_columns:
            print("\nAll required columns are present!")
            return True
        
        print(f"\nMissing columns: {missing_columns}")
        
        # Add missing columns
        for col_name in missing_columns:
            if col_name == 'id':
                # Skip ID column as it should already exist
                continue
                
            print(f"Adding column '{col_name}'...")
            
            if col_name == 'email_validated':
                cursor.execute("""
                    ALTER TABLE users 
                    ADD COLUMN email_validated BOOLEAN NOT NULL DEFAULT FALSE
                """)
            elif col_name == 'last_login_at':
                cursor.execute("""
                    ALTER TABLE users 
                    ADD COLUMN last_login_at TIMESTAMP WITH TIME ZONE
                """)
            elif col_name == 'created_at':
                cursor.execute("""
                    ALTER TABLE users 
                    ADD COLUMN created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                """)
            elif col_name == 'role':
                cursor.execute("""
                    ALTER TABLE users 
                    ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'user'
                """)
            else:
                print(f"  Warning: Don't know how to add column '{col_name}', skipping...")
                continue
            
            print(f"  Successfully added '{col_name}'")
        
        conn.commit()
        
        # Verify all columns are now present
        print("\nVerifying schema after changes...")
        updated_columns = get_table_columns(cursor, 'users')
        updated_column_names = [col[0] for col in updated_columns]
        
        print(f"Updated columns in 'users' table:")
        for col in updated_columns:
            print(f"  - {col[0]} ({col[1]}) {'NULL' if col[2] == 'YES' else 'NOT NULL'}")
        
        # Check if all expected columns are now present
        still_missing = []
        for col_name in expected_columns.keys():
            if col_name not in updated_column_names:
                still_missing.append(col_name)
        
        if still_missing:
            print(f"\nWarning: Some columns are still missing: {still_missing}")
            return False
        else:
            print(f"\nSuccess: All required columns are now present!")
            return True
            
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def main():
    """Main function."""
    print("Complete Database Schema Fix Script")
    print("=" * 40)
    
    success = add_missing_columns()
    
    if success:
        print("\nDatabase schema fix completed successfully!")
        print("You can now run the API server.")
    else:
        print("\nDatabase schema fix failed!")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()