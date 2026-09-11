"""
Generates a structured feedback report for the representative
by sending the full call transcript to Azure OpenAI Chat API.
"""

import json
import os

from openai import AzureOpenAI


PARAMETERS = [
    {
        "key": "greeting",
        "label": "Greeting & Opening",
        "description": "Did the representative greet the customer professionally and introduce themselves clearly?",
    },
    {
        "key": "tone",
        "label": "Tone & Empathy",
        "description": "Was the representative's tone calm, polite, and empathetic throughout the call?",
    },
    {
        "key": "problem_solving",
        "label": "Problem Solving",
        "description": "Did the representative understand the issue and provide clear, accurate, and helpful responses?",
    },
    {
        "key": "communication",
        "label": "Communication Clarity",
        "description": "Did the representative communicate in clear English, avoid jargon, and explain things in a way the customer could understand?",
    },
    {
        "key": "closing",
        "label": "Closing & Wrap-up",
        "description": "Did the representative summarise the resolution, ask if there was anything else, and close the call politely?",
    },
]


def _build_evaluation_prompt(transcript: str, scenario_description: str) -> str:
    parameters_text = "\n".join(
        f"{i + 1}. {p['label']}: {p['description']}"
        for i, p in enumerate(PARAMETERS)
    )

    return f"""You are a quality assurance evaluator for a bank customer care training programme.

You will be given a transcript of a mock call between a trainee representative (REP) and a simulated customer (BOT).

Scenario context: {scenario_description}

Evaluate the representative's performance on the following 5 parameters:
{parameters_text}

For each parameter, provide:
- score: an integer from 1 to 10
- comment: one concise sentence explaining the score (max 20 words)

Respond ONLY with a valid JSON object in exactly this format, no extra text:
{{
  "greeting":       {{"score": <1-10>, "comment": "<one sentence>"}},
  "tone":           {{"score": <1-10>, "comment": "<one sentence>"}},
  "problem_solving":{{"score": <1-10>, "comment": "<one sentence>"}},
  "communication":  {{"score": <1-10>, "comment": "<one sentence>"}},
  "closing":        {{"score": <1-10>, "comment": "<one sentence>"}}
}}

Call transcript:
{transcript}"""


def generate_feedback(transcript: str, scenario_description: str) -> dict:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_CHAT_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )

    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
    if not deployment:
        raise ValueError("Set AZURE_OPENAI_CHAT_DEPLOYMENT in .env")

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "user", "content": _build_evaluation_prompt(transcript, scenario_description)}
        ],
        temperature=0.2,
        max_tokens=400,
    )

    raw = response.choices[0].message.content.strip()
    scores = json.loads(raw)

    total = sum(v["score"] for v in scores.values())
    overall = round(total / len(PARAMETERS), 1)

    return {
        "parameters": [
            {
                "key": p["key"],
                "label": p["label"],
                "score": scores[p["key"]]["score"],
                "comment": scores[p["key"]]["comment"],
            }
            for p in PARAMETERS
        ],
        "overall": overall,
    }
