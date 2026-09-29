import os

# Must be set before anything imports src.config: Settings() reads the
# environment (which overrides .env) at import time.
os.environ["POSTGRES_DB"] = "dmp_test"

from types import SimpleNamespace  # noqa: E402

import psycopg  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.config import settings  # noqa: E402


def _ensure_test_database() -> None:
    with psycopg.connect(
        host="localhost",
        port=settings.database_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname="postgres",
        autocommit=True,
    ) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", ("dmp_test",)).fetchone()
        if not exists:
            conn.execute("CREATE DATABASE dmp_test")


_ensure_test_database()

from src.database import Base, SessionLocal, engine  # noqa: E402
from src.main import app  # noqa: E402
from src.models import EmailConnection, EmailProvider, User  # noqa: E402
from src.models import Session as SessionModel  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_schema():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def connected(db, client):
    user = User()
    db.add(user)
    db.flush()
    session = SessionModel(user_id=user.id, remember_me=True)
    connection = EmailConnection(
        user_id=user.id,
        provider=EmailProvider.gmail,
        provider_account_id="google-sub-1",
        provider_email="me@example.com",
    )
    db.add_all([session, connection])
    db.commit()
    client.cookies.set("dmp_session", str(session.id))
    return SimpleNamespace(user=user, session=session, connection=connection)
