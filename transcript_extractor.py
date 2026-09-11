"""
transcript_extractor.py
=======================
Uses GPT-4.1 to extract valid customer questions from redacted historical
call transcripts and organise them into call types and topics.

What is EXCLUDED (per resume spec):
  - Greeting / pleasantries  (e.g. "Hello", "How are you?")
  - Authentication exchanges  (e.g. "Can you confirm your DOB?")
  - Call-closure lines        (e.g. "Is there anything else?", "Goodbye")

What is INCLUDED:
  - Substantive customer questions about their issue
  - Follow-up questions about process, timelines, next steps

Usage (standalone)
------------------
    python transcript_extractor.py sample_transcripts/

Usage (imported)
----------------
    from transcript_extractor import extract_from_text, extract_from_file

    result = extract_from_text(raw_transcript_text, known_call_types)
    # result is an ExtractionResult with .call_type, .topic, .questions
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import AzureOpenAI, OpenAI

logger = logging.getLogger("extractor")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Structured output from one transcript extraction."""
    call_type: str                   # matched key from known_call_types, or new suggestion
    call_type_title: str
    topic: str                       # sub-topic label, e.g. "Card Cloning"
    questions: list[str] = field(default_factory=list)
    raw_response: str = ""           # full JSON string returned by the model


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a training-data curator for a bank customer-support training platform.

    Your job is to read a redacted customer-service call transcript and extract
    ONLY the substantive questions the customer asked about their banking issue.

    EXCLUDE the following categories of text completely:
      - Greetings and small talk  ("Hello", "Good morning", "How are you?")
      - Authentication exchanges  ("Can you confirm your account number?",
                                   "What is your date of birth?", PIN or OTP prompts)
      - Call-closure interactions ("Is there anything else?", "Have a great day",
                                   "Thank you for calling", "Goodbye")
      - Agent statements or instructions (only customer speech is relevant)
      - Filler lines with no question ("I see", "Okay", "Right")

    INCLUDE only questions that:
      - Ask about the customer's banking issue, problem, or request
      - Ask about processes, timelines, policies, next steps, or options
      - Express concern or urgency in the form of a question

    Also identify:
      - call_type_key: the best matching key from the provided list, or propose a new snake_case key
      - call_type_title: human-readable title
      - topic: a short sub-topic label (2–5 words) describing the specific issue within that call type

    Respond ONLY with a valid JSON object in exactly this format:
    {
      "call_type_key":   "<snake_case_key>",
      "call_type_title": "<Human Readable Title>",
      "topic":           "<Short Sub-Topic Label>",
      "questions": [
        "<question 1>",
        "<question 2>"
      ]
    }

    If the transcript contains fewer than 2 extractable questions, return an empty questions array.
    Do not add explanatory text outside the JSON object.
""").strip()


def _build_user_prompt(transcript: str, known_call_types: list[str]) -> str:
    ct_list = "\n".join(f"  - {k}" for k in known_call_types) if known_call_types else "  (none provided)"
    return (
        f"Known call-type keys (prefer these if they match):\n{ct_list}\n\n"
        f"Transcript:\n{transcript.strip()}"
    )


# ── Client factory ─────────────────────────────────────────────────────────────

def _make_client() -> tuple[AzureOpenAI | OpenAI, str]:
    """
    Return (client, deployment_name).
    Prefers Azure OpenAI if AZURE_OPENAI_CHAT_ENDPOINT is set,
    falls back to standard OpenAI otherwise.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT", "").strip()
    azure_key      = os.getenv("AZURE_OPENAI_CHAT_KEY", "").strip()
    azure_deploy   = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
    api_version    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()

    if azure_endpoint and azure_key and azure_deploy:
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=api_version,
        )
        return client, azure_deploy

    # Fall back to standard OpenAI with gpt-4.1
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise ValueError(
            "No OpenAI credentials found. "
            "Set AZURE_OPENAI_CHAT_ENDPOINT + AZURE_OPENAI_CHAT_KEY + AZURE_OPENAI_CHAT_DEPLOYMENT "
            "or OPENAI_API_KEY in your .env file."
        )
    client = OpenAI(api_key=openai_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    return client, model


# ── Core extraction function ───────────────────────────────────────────────────

def extract_from_text(
    transcript: str,
    known_call_types: Optional[list[str]] = None,
) -> ExtractionResult:
    """
    Send *transcript* to GPT-4.1 and return an ExtractionResult.

    Parameters
    ----------
    transcript:        Raw or redacted call transcript text.
    known_call_types:  List of existing call-type keys so the model can pick
                       the best match rather than always inventing new ones.
    """
    if not transcript.strip():
        raise ValueError("Transcript is empty.")

    known_call_types = known_call_types or []

    client, deployment = _make_client()

    logger.info("Sending transcript to GPT-4.1 for extraction (%d chars)", len(transcript))

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(transcript, known_call_types)},
        ],
        temperature=0.1,      # low temperature for consistent structured output
        max_tokens=1200,
    )

    raw = response.choices[0].message.content.strip()
    logger.debug("Raw model response: %s", raw)

    # Strip markdown code fences if the model wraps JSON in ```json ... ```
    raw_json = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw_json = re.sub(r"\s*```$", "", raw_json)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}\n\nRaw response:\n{raw}") from exc

    questions = data.get("questions", [])
    if not isinstance(questions, list):
        questions = []

    # Clean up whitespace in each question
    questions = [q.strip() for q in questions if isinstance(q, str) and q.strip()]

    return ExtractionResult(
        call_type=data.get("call_type_key", "unknown"),
        call_type_title=data.get("call_type_title", "Unknown"),
        topic=data.get("topic", "General"),
        questions=questions,
        raw_response=raw,
    )


def extract_from_file(
    path: str | Path,
    known_call_types: Optional[list[str]] = None,
) -> ExtractionResult:
    """Read a transcript file and extract questions from it."""
    text = Path(path).read_text(encoding="utf-8")
    logger.info("Loaded transcript file: %s (%d chars)", path, len(text))
    return extract_from_text(text, known_call_types)


# ── Batch extraction ──────────────────────────────────────────────────────────

@dataclass
class BatchExtractionResult:
    """Summary of a batch run over multiple transcript files."""
    total_files:     int = 0
    successful:      int = 0
    failed:          int = 0
    total_questions: int = 0
    results:         list[tuple[str, ExtractionResult]] = field(default_factory=list)
    errors:          list[tuple[str, str]]              = field(default_factory=list)


def extract_from_directory(
    directory: str | Path,
    known_call_types: Optional[list[str]] = None,
    extensions: tuple[str, ...] = (".txt", ".md"),
) -> BatchExtractionResult:
    """
    Process every transcript file in *directory* and return a BatchExtractionResult.

    Only files with extensions in *extensions* are processed.
    """
    directory = Path(directory)
    files = [f for f in directory.iterdir() if f.suffix.lower() in extensions and f.is_file()]
    files.sort()

    batch = BatchExtractionResult(total_files=len(files))
    logger.info("Batch extraction: found %d transcript files in %s", len(files), directory)

    for file_path in files:
        try:
            result = extract_from_file(file_path, known_call_types)
            batch.results.append((str(file_path), result))
            batch.successful      += 1
            batch.total_questions += len(result.questions)
            logger.info(
                "  ✓ %s  →  call_type=%r  topic=%r  questions=%d",
                file_path.name, result.call_type, result.topic, len(result.questions),
            )
        except Exception as exc:                                    # noqa: BLE001
            batch.failed += 1
            batch.errors.append((str(file_path), str(exc)))
            logger.warning("  ✗ %s  →  %s", file_path.name, exc)

    logger.info(
        "Batch complete: %d/%d succeeded, %d questions extracted",
        batch.successful, batch.total_files, batch.total_questions,
    )
    return batch


# ── DB persistence helper ─────────────────────────────────────────────────────

def save_extraction_to_db(
    result: ExtractionResult,
    db,                          # SQLAlchemy Session
) -> dict:
    """
    Persist an ExtractionResult into the database.

    - Upserts the CallType.
    - Creates the Topic if it doesn't exist.
    - Inserts each Question (skips exact duplicates within the same call_type).

    Returns a summary dict with counts.
    """
    from models import CallType, Question, Topic  # local import to avoid circular deps

    # ── CallType ────────────────────────────────────────────────────────────
    call_type_row = db.query(CallType).filter_by(key=result.call_type).first()
    if call_type_row is None:
        call_type_row = CallType(
            key=result.call_type,
            title=result.call_type_title,
        )
        db.add(call_type_row)
        db.flush()  # get the ID without committing

    # ── Topic ────────────────────────────────────────────────────────────────
    topic_row = (
        db.query(Topic)
        .filter_by(call_type_id=call_type_row.id, name=result.topic)
        .first()
    )
    if topic_row is None:
        topic_row = Topic(call_type_id=call_type_row.id, name=result.topic)
        db.add(topic_row)
        db.flush()

    # ── Questions ─────────────────────────────────────────────────────────────
    existing_texts = {
        q.text
        for q in db.query(Question).filter_by(call_type_id=call_type_row.id).all()
    }

    new_count = 0
    for i, text in enumerate(result.questions):
        if text in existing_texts:
            continue
        db.add(Question(
            call_type_id=call_type_row.id,
            topic_id=topic_row.id,
            text=text,
            source="extracted",
            order=i,
        ))
        existing_texts.add(text)
        new_count += 1

    db.commit()

    return {
        "call_type": result.call_type,
        "topic":     result.topic,
        "inserted":  new_count,
        "skipped":   len(result.questions) - new_count,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python transcript_extractor.py <transcript_file_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])

    # Default known call types from the existing scenarios
    KNOWN = [
        "fraudulent_transaction",
        "loan_application",
        "account_locked",
        "fixed_deposit_inquiry",
        "credit_card_dispute",
    ]

    if target.is_dir():
        batch = extract_from_directory(target, known_call_types=KNOWN)
        print(f"\nBatch summary: {batch.successful}/{batch.total_files} files, "
              f"{batch.total_questions} questions extracted.")
        for file_path, r in batch.results:
            print(f"\n  File : {Path(file_path).name}")
            print(f"  Type : {r.call_type_title} ({r.call_type})")
            print(f"  Topic: {r.topic}")
            for i, q in enumerate(r.questions, 1):
                print(f"    {i}. {q}")
        for file_path, err in batch.errors:
            print(f"\n  ERROR in {Path(file_path).name}: {err}")
    else:
        r = extract_from_file(target, known_call_types=KNOWN)
        print(f"\nCall Type : {r.call_type_title} ({r.call_type})")
        print(f"Topic     : {r.topic}")
        print(f"Questions : {len(r.questions)}")
        for i, q in enumerate(r.questions, 1):
            print(f"  {i}. {q}")
