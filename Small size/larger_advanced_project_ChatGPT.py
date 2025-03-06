"""
This module is part of a larger software project.
It includes:
- Logging and configuration management
- Database interaction with SQLite
- API request handling
- Data processing utilities
"""

import json
import logging
import sqlite3
import requests
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
            logger.info("Configuration loaded successfully from %s", file_path)
            return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Error loading config file: %s", e)
        return {}

def save_config(file_path: str, config: Dict[str, Any]) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(config, file, indent=4)
            logger.info("Configuration saved to %s", file_path)
    except IOError as e:
        logger.error("Error saving config file: %s", e)

# Database Interaction (SQLite)
def initialize_db(db_path: str):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    ''')
    connection.commit()
    connection.close()
    logger.info("Database initialized at %s", db_path)

def add_user(db_path: str, name: str, email: str) -> None:
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
        connection.commit()
        logger.info("User added: %s", name)
    except sqlite3.IntegrityError:
        logger.error("Email already exists: %s", email)
    finally:
        connection.close()

def get_users(db_path: str) -> List[Dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT id, name, email FROM users")
    users = [{"id": row[0], "name": row[1], "email": row[2]} for row in cursor.fetchall()]
    connection.close()
    return users

# API Request Handling
def fetch_data_from_api(url: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error("API request failed: %s", e)
        return None

# Example usage
if __name__ == "__main__":
    config_path = "config.json"
    db_path = "users.db"
    
    # Config management
    sample_config = {"app_name": "EnterpriseApp", "version": "2.0.0", "debug": True}
    save_config(config_path, sample_config)
    loaded_config = load_config(config_path)
    print("Loaded Configuration:", loaded_config)
    
    # Database operations
    initialize_db(db_path)
    add_user(db_path, "Alice Johnson", "alice@example.com")
    add_user(db_path, "Bob Smith", "bob@example.com")
    print("Users:", get_users(db_path))
    
    # API interaction
    api_url = "https://jsonplaceholder.typicode.com/posts/1"
    api_data = fetch_data_from_api(api_url)
    if api_data:
        print("API Data:", api_data)
