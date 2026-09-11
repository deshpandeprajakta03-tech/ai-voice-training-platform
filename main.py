"""
main.py
=======
FastAPI application for the AI Voice Training Platform.

Endpoints
---------
Voice / UI
  GET  /                          – serve the training simulator UI
  GET  /favicon.ico               – suppress browser 404
  GET  /scenarios                 – list all scenarios (for the voice UI dropdown)
  WS   /ws/audio                  – real-time voice WebSocket bridge
  POST /feedback                  – generate post-call feedback report

Training Levels
  GET  /api/levels                – list all training levels
  POST /api/levels                – create a training level
  PUT  /api/levels/{id}           – update a training level
  DELETE /api/levels/{id}         – delete a training level

Call Types
  GET  /api/call-types            – list all call types
  POST /api/call-types            – create a call type
  PUT  /api/call-types/{id}       – update a call type
  DELETE /api/call-types/{id}     – delete a call type

Questions
  GET  /api/questions             – list questions (filter by call_type_id or topic_id)
  POST /api/questions             – create a question
  PUT  /api/questions/{id}        – update a question
  DELETE /api/questions/{id}      – delete a question

Representatives
  GET  /api/representatives       – list all representatives
  POST /api/representatives       – create a representative
  PUT  /api/representatives/{id}  – update a representative
  DELETE /api/representatives/{id}– delete a representative

Assignments
  GET  /api/assignments           – list assignments (filter by rep_id or status)
  POST /api/assignments           – create an assignment
  PUT  /api/assignments/{id}      – update assignment status
  DELETE /api/assignments/{id}    – delete an assignment

Feedback Results
  GET  /api/feedback-results      – list stored feedback results (filter by rep_id)
  GET  /api/feedback-results/{id} – get a single feedback result
  DELETE /api/feedback-results/{id} – delete a feedback result

Transcript Extraction
  POST /api/extract               – extract questions from a pasted transcript via GPT-4.1
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from azure_bridge import get_last_transcript, voice_socket
from feedback import generate_feedback
from models import (
    Assignment,
    CallType,
    FeedbackResult,
    Question,
    Representative,
    Topic,
    TrainingLevel,
    get_db,
    init_db,
)
from scenarios import DEFAULT_SCENARIO, SCENARIOS, scenario_summary
from transcript_extractor import ExtractionResult, extract_from_text, save_extraction_to_db

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env", override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# ── App ───────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialised.")
    yield


app = FastAPI(
    title="AI Voice Training Platform",
    description="Simulate realistic customer-support training calls powered by Azure OpenAI.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class LevelCreate(BaseModel):
    name: str
    description: Optional[str] = None
    order: int = 0


class LevelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    order: Optional[int] = None


class CallTypeCreate(BaseModel):
    key: str
    title: str
    description: Optional[str] = None


class CallTypeUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None


class QuestionCreate(BaseModel):
    call_type_id: int
    topic_id: Optional[int] = None
    text: str
    source: str = "manual"
    order: int = 0


class QuestionUpdate(BaseModel):
    text: Optional[str] = None
    topic_id: Optional[int] = None
    order: Optional[int] = None


class RepCreate(BaseModel):
    name: str
    email: EmailStr


class RepUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None


class AssignmentCreate(BaseModel):
    rep_id: int
    level_id: int
    call_type_id: int


class AssignmentUpdate(BaseModel):
    status: str  # "pending" | "in_progress" | "completed"


class ExtractRequest(BaseModel):
    transcript: str
    save_to_db: bool = False


# ── Utility ───────────────────────────────────────────────────────────────────

def _utcnow():
    return datetime.now(timezone.utc)


def _not_found(entity: str, id: int):
    raise HTTPException(status_code=404, detail=f"{entity} with id={id} not found.")


# ══════════════════════════════════════════════════════════════════════════════
# Voice / UI routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/scenarios", tags=["Voice UI"])
async def list_scenarios():
    """Return all scenarios for the voice UI dropdown."""
    return {
        key: {"title": s["title"], "description": s["description"]}
        for key, s in SCENARIOS.items()
    }


@app.get("/scenarios/summary", tags=["Voice UI"])
async def scenarios_summary():
    """Return enriched scenario summaries including call_type, topic, level."""
    return scenario_summary()


@app.post("/feedback", tags=["Voice UI"])
async def feedback():
    """Generate a scored feedback report for the most recently completed call."""
    transcript, scenario_description = get_last_transcript()
    if not transcript:
        return JSONResponse(
            {"error": "No call transcript available. Complete a call first."},
            status_code=400,
        )
    try:
        report = generate_feedback(transcript, scenario_description)
        return JSONResponse(report)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.websocket("/ws/audio")
async def ws_audio(
    browser: WebSocket,
    scenario: str = Query(default=DEFAULT_SCENARIO),
):
    await voice_socket(browser, scenario)


# ══════════════════════════════════════════════════════════════════════════════
# Training Levels
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/levels", tags=["Training Levels"])
def get_levels(db: Session = Depends(get_db)):
    """List all training levels ordered by difficulty."""
    levels = db.query(TrainingLevel).order_by(TrainingLevel.order).all()
    return [
        {"id": l.id, "name": l.name, "description": l.description, "order": l.order}
        for l in levels
    ]


@app.post("/api/levels", tags=["Training Levels"], status_code=201)
def create_level(body: LevelCreate, db: Session = Depends(get_db)):
    """Create a new training level."""
    if db.query(TrainingLevel).filter_by(name=body.name).first():
        raise HTTPException(status_code=409, detail=f"Level '{body.name}' already exists.")
    level = TrainingLevel(name=body.name, description=body.description, order=body.order)
    db.add(level)
    db.commit()
    db.refresh(level)
    return {"id": level.id, "name": level.name, "description": level.description, "order": level.order}


@app.put("/api/levels/{level_id}", tags=["Training Levels"])
def update_level(level_id: int, body: LevelUpdate, db: Session = Depends(get_db)):
    """Update an existing training level."""
    level = db.get(TrainingLevel, level_id)
    if not level:
        _not_found("TrainingLevel", level_id)
    if body.name is not None:
        level.name = body.name
    if body.description is not None:
        level.description = body.description
    if body.order is not None:
        level.order = body.order
    db.commit()
    db.refresh(level)
    return {"id": level.id, "name": level.name, "description": level.description, "order": level.order}


@app.delete("/api/levels/{level_id}", tags=["Training Levels"], status_code=204)
def delete_level(level_id: int, db: Session = Depends(get_db)):
    """Delete a training level."""
    level = db.get(TrainingLevel, level_id)
    if not level:
        _not_found("TrainingLevel", level_id)
    db.delete(level)
    db.commit()
    return Response(status_code=204)


# ══════════════════════════════════════════════════════════════════════════════
# Call Types
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/call-types", tags=["Call Types"])
def get_call_types(db: Session = Depends(get_db)):
    """List all call types."""
    cts = db.query(CallType).order_by(CallType.title).all()
    return [
        {"id": ct.id, "key": ct.key, "title": ct.title, "description": ct.description}
        for ct in cts
    ]


@app.post("/api/call-types", tags=["Call Types"], status_code=201)
def create_call_type(body: CallTypeCreate, db: Session = Depends(get_db)):
    """Create a new call type."""
    if db.query(CallType).filter_by(key=body.key).first():
        raise HTTPException(status_code=409, detail=f"Call type key '{body.key}' already exists.")
    ct = CallType(key=body.key, title=body.title, description=body.description)
    db.add(ct)
    db.commit()
    db.refresh(ct)
    return {"id": ct.id, "key": ct.key, "title": ct.title, "description": ct.description}


@app.put("/api/call-types/{ct_id}", tags=["Call Types"])
def update_call_type(ct_id: int, body: CallTypeUpdate, db: Session = Depends(get_db)):
    """Update a call type."""
    ct = db.get(CallType, ct_id)
    if not ct:
        _not_found("CallType", ct_id)
    if body.title is not None:
        ct.title = body.title
    if body.description is not None:
        ct.description = body.description
    db.commit()
    db.refresh(ct)
    return {"id": ct.id, "key": ct.key, "title": ct.title, "description": ct.description}


@app.delete("/api/call-types/{ct_id}", tags=["Call Types"], status_code=204)
def delete_call_type(ct_id: int, db: Session = Depends(get_db)):
    """Delete a call type and all its questions."""
    ct = db.get(CallType, ct_id)
    if not ct:
        _not_found("CallType", ct_id)
    db.delete(ct)
    db.commit()
    return Response(status_code=204)


# ══════════════════════════════════════════════════════════════════════════════
# Questions
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/questions", tags=["Questions"])
def get_questions(
    call_type_id: Optional[int] = Query(default=None),
    topic_id: Optional[int] = Query(default=None),
    source: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """
    List questions with optional filters.
    - call_type_id: filter by call type
    - topic_id: filter by topic
    - source: 'manual' or 'extracted'
    """
    q = db.query(Question)
    if call_type_id is not None:
        q = q.filter(Question.call_type_id == call_type_id)
    if topic_id is not None:
        q = q.filter(Question.topic_id == topic_id)
    if source is not None:
        q = q.filter(Question.source == source)
    questions = q.order_by(Question.call_type_id, Question.order).all()
    return [
        {
            "id":           qn.id,
            "call_type_id": qn.call_type_id,
            "topic_id":     qn.topic_id,
            "text":         qn.text,
            "source":       qn.source,
            "order":        qn.order,
            "created_at":   qn.created_at.isoformat() if qn.created_at else None,
        }
        for qn in questions
    ]


@app.post("/api/questions", tags=["Questions"], status_code=201)
def create_question(body: QuestionCreate, db: Session = Depends(get_db)):
    """Create a new training question."""
    if not db.get(CallType, body.call_type_id):
        raise HTTPException(status_code=404, detail=f"CallType id={body.call_type_id} not found.")
    qn = Question(
        call_type_id=body.call_type_id,
        topic_id=body.topic_id,
        text=body.text,
        source=body.source,
        order=body.order,
    )
    db.add(qn)
    db.commit()
    db.refresh(qn)
    return {"id": qn.id, "call_type_id": qn.call_type_id, "text": qn.text, "source": qn.source}


@app.put("/api/questions/{q_id}", tags=["Questions"])
def update_question(q_id: int, body: QuestionUpdate, db: Session = Depends(get_db)):
    """Update a question's text, topic, or order."""
    qn = db.get(Question, q_id)
    if not qn:
        _not_found("Question", q_id)
    if body.text is not None:
        qn.text = body.text
    if body.topic_id is not None:
        qn.topic_id = body.topic_id
    if body.order is not None:
        qn.order = body.order
    db.commit()
    db.refresh(qn)
    return {"id": qn.id, "text": qn.text, "topic_id": qn.topic_id, "order": qn.order}


@app.delete("/api/questions/{q_id}", tags=["Questions"], status_code=204)
def delete_question(q_id: int, db: Session = Depends(get_db)):
    """Delete a question."""
    qn = db.get(Question, q_id)
    if not qn:
        _not_found("Question", q_id)
    db.delete(qn)
    db.commit()
    return Response(status_code=204)


# ══════════════════════════════════════════════════════════════════════════════
# Representatives
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/representatives", tags=["Representatives"])
def get_representatives(db: Session = Depends(get_db)):
    """List all trainees/representatives."""
    reps = db.query(Representative).order_by(Representative.name).all()
    return [
        {
            "id":         r.id,
            "name":       r.name,
            "email":      r.email,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in reps
    ]


@app.post("/api/representatives", tags=["Representatives"], status_code=201)
def create_representative(body: RepCreate, db: Session = Depends(get_db)):
    """Register a new representative."""
    if db.query(Representative).filter_by(email=body.email).first():
        raise HTTPException(status_code=409, detail=f"Email '{body.email}' already registered.")
    rep = Representative(name=body.name, email=body.email)
    db.add(rep)
    db.commit()
    db.refresh(rep)
    return {"id": rep.id, "name": rep.name, "email": rep.email}


@app.put("/api/representatives/{rep_id}", tags=["Representatives"])
def update_representative(rep_id: int, body: RepUpdate, db: Session = Depends(get_db)):
    """Update a representative's details."""
    rep = db.get(Representative, rep_id)
    if not rep:
        _not_found("Representative", rep_id)
    if body.name is not None:
        rep.name = body.name
    if body.email is not None:
        rep.email = body.email
    db.commit()
    db.refresh(rep)
    return {"id": rep.id, "name": rep.name, "email": rep.email}


@app.delete("/api/representatives/{rep_id}", tags=["Representatives"], status_code=204)
def delete_representative(rep_id: int, db: Session = Depends(get_db)):
    """Delete a representative and all their assignments."""
    rep = db.get(Representative, rep_id)
    if not rep:
        _not_found("Representative", rep_id)
    db.delete(rep)
    db.commit()
    return Response(status_code=204)


# ══════════════════════════════════════════════════════════════════════════════
# Assignments
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/assignments", tags=["Assignments"])
def get_assignments(
    rep_id: Optional[int] = Query(default=None),
    status: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """List assignments with optional filters by rep or status."""
    q = db.query(Assignment)
    if rep_id is not None:
        q = q.filter(Assignment.rep_id == rep_id)
    if status is not None:
        q = q.filter(Assignment.status == status)
    assignments = q.order_by(Assignment.assigned_at.desc()).all()
    return [
        {
            "id":           a.id,
            "rep_id":       a.rep_id,
            "rep_name":     a.rep.name if a.rep else None,
            "level_id":     a.level_id,
            "level_name":   a.level.name if a.level else None,
            "call_type_id": a.call_type_id,
            "call_type":    a.call_type.title if a.call_type else None,
            "status":       a.status,
            "assigned_at":  a.assigned_at.isoformat() if a.assigned_at else None,
            "completed_at": a.completed_at.isoformat() if a.completed_at else None,
        }
        for a in assignments
    ]


@app.post("/api/assignments", tags=["Assignments"], status_code=201)
def create_assignment(body: AssignmentCreate, db: Session = Depends(get_db)):
    """Assign a representative to a training level and call type."""
    if not db.get(Representative, body.rep_id):
        raise HTTPException(status_code=404, detail=f"Representative id={body.rep_id} not found.")
    if not db.get(TrainingLevel, body.level_id):
        raise HTTPException(status_code=404, detail=f"TrainingLevel id={body.level_id} not found.")
    if not db.get(CallType, body.call_type_id):
        raise HTTPException(status_code=404, detail=f"CallType id={body.call_type_id} not found.")

    existing = (
        db.query(Assignment)
        .filter_by(rep_id=body.rep_id, level_id=body.level_id, call_type_id=body.call_type_id)
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="Assignment already exists.")

    a = Assignment(rep_id=body.rep_id, level_id=body.level_id, call_type_id=body.call_type_id)
    db.add(a)
    db.commit()
    db.refresh(a)
    return {
        "id":           a.id,
        "rep_id":       a.rep_id,
        "level_id":     a.level_id,
        "call_type_id": a.call_type_id,
        "status":       a.status,
        "assigned_at":  a.assigned_at.isoformat() if a.assigned_at else None,
    }


@app.put("/api/assignments/{assignment_id}", tags=["Assignments"])
def update_assignment(assignment_id: int, body: AssignmentUpdate, db: Session = Depends(get_db)):
    """Update assignment status (pending → in_progress → completed)."""
    valid = {"pending", "in_progress", "completed"}
    if body.status not in valid:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid}.")
    a = db.get(Assignment, assignment_id)
    if not a:
        _not_found("Assignment", assignment_id)
    a.status = body.status
    if body.status == "completed":
        a.completed_at = _utcnow()
    db.commit()
    db.refresh(a)
    return {"id": a.id, "status": a.status, "completed_at": a.completed_at.isoformat() if a.completed_at else None}


@app.delete("/api/assignments/{assignment_id}", tags=["Assignments"], status_code=204)
def delete_assignment(assignment_id: int, db: Session = Depends(get_db)):
    """Delete an assignment."""
    a = db.get(Assignment, assignment_id)
    if not a:
        _not_found("Assignment", assignment_id)
    db.delete(a)
    db.commit()
    return Response(status_code=204)


# ══════════════════════════════════════════════════════════════════════════════
# Feedback Results
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/feedback-results", tags=["Feedback Results"])
def get_feedback_results(
    rep_id: Optional[int] = Query(default=None),
    scenario_key: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """List stored feedback results with optional filters."""
    q = db.query(FeedbackResult)
    if rep_id is not None:
        q = q.filter(FeedbackResult.rep_id == rep_id)
    if scenario_key is not None:
        q = q.filter(FeedbackResult.scenario_key == scenario_key)
    results = q.order_by(FeedbackResult.created_at.desc()).all()
    return [_format_feedback_result(r) for r in results]


@app.get("/api/feedback-results/{result_id}", tags=["Feedback Results"])
def get_feedback_result(result_id: int, db: Session = Depends(get_db)):
    """Get a single feedback result by ID."""
    r = db.get(FeedbackResult, result_id)
    if not r:
        _not_found("FeedbackResult", result_id)
    return _format_feedback_result(r)


@app.delete("/api/feedback-results/{result_id}", tags=["Feedback Results"], status_code=204)
def delete_feedback_result(result_id: int, db: Session = Depends(get_db)):
    """Delete a feedback result."""
    r = db.get(FeedbackResult, result_id)
    if not r:
        _not_found("FeedbackResult", result_id)
    db.delete(r)
    db.commit()
    return Response(status_code=204)


def _format_feedback_result(r: FeedbackResult) -> dict:
    return {
        "id":             r.id,
        "rep_id":         r.rep_id,
        "scenario_key":   r.scenario_key,
        "scenario_title": r.scenario_title,
        "overall":        r.overall,
        "parameters": {
            "greeting":        {"score": r.score_greeting,        "comment": r.comment_greeting},
            "tone":            {"score": r.score_tone,            "comment": r.comment_tone},
            "problem_solving": {"score": r.score_problem_solving, "comment": r.comment_problem_solving},
            "communication":   {"score": r.score_communication,   "comment": r.comment_communication},
            "closing":         {"score": r.score_closing,         "comment": r.comment_closing},
        },
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Transcript Extraction
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/extract", tags=["Transcript Extraction"])
def extract_transcript(body: ExtractRequest, db: Session = Depends(get_db)):
    """
    Extract valid customer questions from a pasted transcript using GPT-4.1.

    Excludes greetings, authentication exchanges, and call-closure lines.
    Optionally saves extracted questions to the database when save_to_db=true.
    """
    known_keys = [ct.key for ct in db.query(CallType).all()]

    try:
        result: ExtractionResult = extract_from_text(body.transcript, known_call_types=known_keys)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    response = {
        "call_type":       result.call_type,
        "call_type_title": result.call_type_title,
        "topic":           result.topic,
        "questions":       result.questions,
        "question_count":  len(result.questions),
        "saved":           False,
    }

    if body.save_to_db and result.questions:
        try:
            summary = save_extraction_to_db(result, db)
            response["saved"]   = True
            response["inserted"] = summary["inserted"]
            response["skipped"]  = summary["skipped"]
        except Exception as exc:
            logger.warning("Failed to save extraction to DB: %s", exc)
            response["save_error"] = str(exc)

    return response


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
