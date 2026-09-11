"""
SQLAlchemy ORM models and database setup for the AI Voice Training Platform.

Tables
------
training_levels        – difficulty tiers (Beginner / Intermediate / Advanced)
call_types             – categories of customer calls (e.g. Fraud, Loan, Dispute)
topics                 – sub-topics within a call type
questions              – individual customer questions linked to a call_type + topic
representatives        – trainees who use the platform
assignments            – which rep is assigned to which level / call_type
feedback_results       – per-call scored feedback stored after each training session
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ── Database URL ─────────────────────────────────────────────────────────────
# Defaults to a local SQLite file; override via DATABASE_URL in .env.
_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./training_platform.db")

engine = create_engine(
    _DB_URL,
    # Needed for SQLite to allow multi-threaded access from FastAPI
    connect_args={"check_same_thread": False} if _DB_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Base ─────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Helpers ──────────────────────────────────────────────────────────────────
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Models ───────────────────────────────────────────────────────────────────

class TrainingLevel(Base):
    """Difficulty tiers used to gate scenario complexity."""
    __tablename__ = "training_levels"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(64), unique=True, nullable=False)        # e.g. "Beginner"
    description = Column(Text, nullable=True)
    order       = Column(Integer, default=0)                             # sort order

    # Relationships
    assignments = relationship("Assignment", back_populates="level", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<TrainingLevel id={self.id} name={self.name!r}>"


class CallType(Base):
    """High-level category of a customer call (mirrors scenario keys)."""
    __tablename__ = "call_types"

    id          = Column(Integer, primary_key=True, index=True)
    key         = Column(String(64), unique=True, nullable=False)        # machine key, e.g. "fraud"
    title       = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)

    # Relationships
    topics      = relationship("Topic",      back_populates="call_type", cascade="all, delete-orphan")
    questions   = relationship("Question",   back_populates="call_type", cascade="all, delete-orphan")
    assignments = relationship("Assignment", back_populates="call_type", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<CallType id={self.id} key={self.key!r}>"


class Topic(Base):
    """Sub-topic within a call type (e.g. 'Card Cloning' inside 'Fraud')."""
    __tablename__ = "topics"

    id           = Column(Integer, primary_key=True, index=True)
    call_type_id = Column(Integer, ForeignKey("call_types.id"), nullable=False)
    name         = Column(String(128), nullable=False)
    description  = Column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("call_type_id", "name", name="uq_topic_name_per_calltype"),)

    # Relationships
    call_type = relationship("CallType",  back_populates="topics")
    questions = relationship("Question",  back_populates="topic",     cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Topic id={self.id} name={self.name!r}>"


class Question(Base):
    """
    A single customer question extracted from transcripts or manually authored.
    Linked to a CallType and optionally a Topic.
    """
    __tablename__ = "questions"

    id           = Column(Integer, primary_key=True, index=True)
    call_type_id = Column(Integer, ForeignKey("call_types.id"), nullable=False)
    topic_id     = Column(Integer, ForeignKey("topics.id"),     nullable=True)
    text         = Column(Text, nullable=False)
    source       = Column(String(32), default="manual")          # "manual" | "extracted"
    order        = Column(Integer, default=0)
    created_at   = Column(DateTime(timezone=True), default=_utcnow)

    # Relationships
    call_type = relationship("CallType", back_populates="questions")
    topic     = relationship("Topic",    back_populates="questions")

    def __repr__(self) -> str:
        return f"<Question id={self.id} call_type_id={self.call_type_id}>"


class Representative(Base):
    """A customer-support trainee who uses the platform."""
    __tablename__ = "representatives"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(128), nullable=False)
    email      = Column(String(256), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    # Relationships
    assignments      = relationship("Assignment",    back_populates="rep", cascade="all, delete-orphan")
    feedback_results = relationship("FeedbackResult", back_populates="rep", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Representative id={self.id} name={self.name!r}>"


class Assignment(Base):
    """
    Which representative is assigned to a specific training level and call type.
    Status: 'pending' | 'in_progress' | 'completed'
    """
    __tablename__ = "assignments"

    id           = Column(Integer, primary_key=True, index=True)
    rep_id       = Column(Integer, ForeignKey("representatives.id"), nullable=False)
    level_id     = Column(Integer, ForeignKey("training_levels.id"), nullable=False)
    call_type_id = Column(Integer, ForeignKey("call_types.id"),      nullable=False)
    status       = Column(String(32), default="pending")
    assigned_at  = Column(DateTime(timezone=True), default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("rep_id", "level_id", "call_type_id", name="uq_assignment"),
    )

    # Relationships
    rep       = relationship("Representative", back_populates="assignments")
    level     = relationship("TrainingLevel",  back_populates="assignments")
    call_type = relationship("CallType",       back_populates="assignments")

    def __repr__(self) -> str:
        return f"<Assignment id={self.id} rep_id={self.rep_id} status={self.status!r}>"


class FeedbackResult(Base):
    """
    Scored feedback report saved after each completed training call.
    Scores are stored as individual columns for easy querying/reporting.
    """
    __tablename__ = "feedback_results"

    id                = Column(Integer, primary_key=True, index=True)
    rep_id            = Column(Integer, ForeignKey("representatives.id"), nullable=True)
    scenario_key      = Column(String(64), nullable=False)
    scenario_title    = Column(String(128), nullable=True)
    transcript        = Column(Text, nullable=True)

    # Individual parameter scores (1–10)
    score_greeting      = Column(Float, nullable=True)
    score_tone          = Column(Float, nullable=True)
    score_problem_solving = Column(Float, nullable=True)
    score_communication = Column(Float, nullable=True)
    score_closing       = Column(Float, nullable=True)

    # Comments
    comment_greeting      = Column(Text, nullable=True)
    comment_tone          = Column(Text, nullable=True)
    comment_problem_solving = Column(Text, nullable=True)
    comment_communication = Column(Text, nullable=True)
    comment_closing       = Column(Text, nullable=True)

    overall    = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    # Relationships
    rep = relationship("Representative", back_populates="feedback_results")

    def __repr__(self) -> str:
        return f"<FeedbackResult id={self.id} overall={self.overall} scenario={self.scenario_key!r}>"


# ── Database initialisation ───────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables and seed default data if the DB is empty."""
    Base.metadata.create_all(bind=engine)
    _seed_defaults()


def _seed_defaults() -> None:
    """Insert default training levels and call types on first run."""
    with SessionLocal() as db:
        # Training levels
        if db.query(TrainingLevel).count() == 0:
            db.add_all([
                TrainingLevel(name="Beginner",     description="Simple, single-issue calls with clear resolution paths.", order=1),
                TrainingLevel(name="Intermediate", description="Multi-issue calls requiring policy knowledge and escalation decisions.", order=2),
                TrainingLevel(name="Advanced",     description="Complex, emotionally charged calls with ambiguous resolutions.", order=3),
            ])

        # Call types (mirroring the existing scenario keys)
        if db.query(CallType).count() == 0:
            call_types = [
                CallType(key="fraudulent_transaction", title="Fraudulent Transaction",
                         description="Unauthorised charges and card-compromise reports."),
                CallType(key="loan_application",       title="Loan Application",
                         description="Status, delays, and expediting personal loan applications."),
                CallType(key="account_locked",         title="Account Locked Out",
                         description="Online banking lockouts and password resets."),
                CallType(key="fixed_deposit_inquiry",  title="Fixed Deposit Inquiry",
                         description="Maturity options, renewal rates, and partial withdrawals."),
                CallType(key="credit_card_dispute",    title="Credit Card Dispute",
                         description="Duplicate charges and chargeback processes."),
            ]
            db.add_all(call_types)

        db.commit()


# ── Dependency for FastAPI ────────────────────────────────────────────────────

def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
