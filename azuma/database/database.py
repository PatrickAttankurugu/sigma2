"""
Database Manager for Azuma AI
Handles database connections, sessions, and operations
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import os
from pathlib import Path

from .models import Base


class DatabaseManager:
    """Manages database connections and sessions"""

    def __init__(self, database_url: str = None):
        """
        Initialize database manager

        Args:
            database_url: Database URL (defaults to SQLite in project root)
        """
        if database_url is None:
            # Default to SQLite in project root
            db_path = Path(__file__).parent.parent.parent / "azuma.db"
            database_url = f"sqlite:///{db_path}"

        # Create engine
        if database_url.startswith("sqlite"):
            # SQLite specific settings
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False  # Set to True for SQL debugging
            )
        else:
            self.engine = create_engine(database_url, echo=False)

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations

        Usage:
            with db_manager.session_scope() as session:
                user = session.query(User).first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Global database manager instance
_db_manager = None


def initialize_database(database_url: str = None) -> DatabaseManager:
    """
    Initialize the global database manager

    Args:
        database_url: Database URL (optional)

    Returns:
        DatabaseManager instance
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    _db_manager.create_tables()
    return _db_manager


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = initialize_database()
    return _db_manager


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database sessions

    Usage in FastAPI:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db_manager = get_db_manager()
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()
