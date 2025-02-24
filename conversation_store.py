import sqlite3
import json
from datetime import datetime
import os

class ConversationStore:
    def __init__(self, db_path="conversations.db"):
        """Initialize conversation store with SQLite database."""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create required tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            """)
            
            conn.commit()
    
    def create_conversation(self, conversation_id):
        """Create a new conversation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO conversations (conversation_id) VALUES (?)",
                    (conversation_id,)
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
    
    def add_message(self, conversation_id, role, content):
        """Add a message to the conversation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    (conversation_id, role, content)
                )
                conn.commit()
                return True
        except sqlite3.Error:
            return False
    
    def get_conversation_history(self, conversation_id, limit=10):
        """Get conversation history in chronological order."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT role, content, timestamp 
                    FROM messages 
                    WHERE conversation_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                    """,
                    (conversation_id, limit)
                )
                messages = cursor.fetchall()
                return [
                    {
                        "role": msg[0],
                        "content": msg[1],
                        "timestamp": msg[2]
                    }
                    for msg in messages
                ]
        except sqlite3.Error:
            return []
    
    def conversation_exists(self, conversation_id):
        """Check if a conversation exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM conversations WHERE conversation_id = ?",
                    (conversation_id,)
                )
                return cursor.fetchone() is not None
        except sqlite3.Error:
            return False
    
    def delete_conversation(self, conversation_id):
        """Delete a conversation and its messages."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Delete messages first due to foreign key constraint
                cursor.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (conversation_id,)
                )
                cursor.execute(
                    "DELETE FROM conversations WHERE conversation_id = ?",
                    (conversation_id,)
                )
                conn.commit()
                return True
        except sqlite3.Error:
            return False 