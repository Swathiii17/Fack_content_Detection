from sqlalchemy import Column, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import uuid
from config import settings

Base = declarative_base()

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content_type = Column(String, nullable=False)
    input_summary = Column(Text)
    final_verdict = Column(String)
    final_confidence = Column(Float)
    final_reasoning = Column(Text)
    all_signals = Column(JSON)
    agent_results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
