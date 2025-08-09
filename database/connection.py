"""
Database connection and session management for the Real Estate Investment System.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator

from config.settings import settings
from database.models import Base


class DatabaseManager:
    """Database connection and session manager."""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._setup_engine()
    
    def _setup_engine(self):
        """Setup database engine with appropriate configuration."""
        
        # For testing, use in-memory SQLite
        if settings.environment == "test":
            database_url = "sqlite:///:memory:"
            engine_kwargs = {
                "echo": settings.debug,
                "connect_args": {"check_same_thread": False},
                "poolclass": StaticPool,
            }
        else:
            # Use SQLite for demo
            database_url = "sqlite:///real_estate_demo.db"
            engine_kwargs = {
                "echo": settings.debug,
                "pool_pre_ping": True,
                "pool_recycle": 3600,
            }
        
        self.engine = create_engine(database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_session_direct(self) -> Session:
        """Get a database session directly (caller must manage lifecycle)."""
        return self.SessionLocal()


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """Dependency function for FastAPI to get database sessions."""
    with db_manager.get_session() as session:
        yield session


def init_database():
    """Initialize the database with tables and any required setup."""
    try:
        # Create tables
        db_manager.create_tables()
        print("✅ Database tables created successfully")
        
        # Add any initial data if needed
        # _add_initial_data()
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise


def _add_initial_data():
    """Add any initial data required for the application."""
    # This can be used to add default market data, configuration, etc.
    pass


# Database health check
def check_database_health() -> bool:
    """Check if the database is accessible and healthy."""
    try:
        with db_manager.get_session() as session:
            # Try a simple query
            from sqlalchemy import text
            session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        print(f"Database health check failed: {e}")
        return False
