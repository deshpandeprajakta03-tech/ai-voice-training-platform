"""
scenarios.py
============
Mock call scenarios for the AI Voice Training Platform.

Each scenario now carries:
  - call_type_key  : matches the CallType.key in the database
  - topic          : sub-topic label for the specific issue
  - level          : suggested training level ("Beginner" | "Intermediate" | "Advanced")
  - questions      : list of dicts with text + topic tag per question

The BOT plays the bank customer and asks these questions to the trainee (agent).
Add, edit, or remove scenarios and questions here freely.
"""

from __future__ import annotations

# ── Scenario registry ─────────────────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {

    "fraudulent_transaction": {
        "title":         "Fraudulent Transaction",
        "call_type_key": "fraudulent_transaction",
        "topic":         "Unauthorised Card Charges",
        "level":         "Intermediate",
        "description": (
            "Customer has noticed two unauthorised transactions on their account "
            "totalling $340 and is panicking that their card has been compromised."
        ),
        "questions": [
            {
                "text": (
                    "Hi, I just checked my account online and I can see two transactions I never made — "
                    "one for $180 at a store in another city and one for $160 on some website. "
                    "I didn't make either of these. What's happening?"
                ),
                "topic": "Unauthorised Card Charges",
            },
            {
                "text": "How is this even possible? I have my card right here with me. Could someone have cloned it?",
                "topic": "Card Cloning",
            },
            {
                "text": "I need those charges reversed immediately. How long will that take?",
                "topic": "Dispute Resolution Timeline",
            },
            {
                "text": (
                    "Should I cancel my card right now? "
                    "What if more unauthorised charges come through while we're talking?"
                ),
                "topic": "Card Cancellation",
            },
            {
                "text": (
                    "Will I receive a new card automatically, and how long will it take to arrive? "
                    "I use this card for everything."
                ),
                "topic": "Replacement Card",
            },
        ],
    },

    "loan_application": {
        "title":         "Loan Application",
        "call_type_key": "loan_application",
        "topic":         "Application Status & Delays",
        "level":         "Intermediate",
        "description": (
            "Customer applied for a personal loan of $15,000 ten days ago "
            "and has not received any update. They need the funds urgently for a medical emergency."
        ),
        "questions": [
            {
                "text": (
                    "I applied for a personal loan of $15,000 about ten days ago and I still haven't "
                    "heard anything. Can you tell me what the status of my application is?"
                ),
                "topic": "Application Status",
            },
            {
                "text": "Ten days feels like a very long time. What is the normal processing time for a personal loan?",
                "topic": "Processing Timeline",
            },
            {
                "text": "This is for a medical emergency — my mother needs surgery. Is there any way to expedite the review?",
                "topic": "Expedited Processing",
            },
            {
                "text": (
                    "My credit score is above 750 and I have been a customer here for over six years. "
                    "Why would there be any delay on my application?"
                ),
                "topic": "Delay Reasons",
            },
            {
                "text": "If the loan is approved today, how quickly can the funds be transferred to my account?",
                "topic": "Fund Disbursement",
            },
        ],
    },

    "account_locked": {
        "title":         "Account Locked Out",
        "call_type_key": "account_locked",
        "topic":         "Online Banking Access",
        "level":         "Beginner",
        "description": (
            "Customer's online banking account has been locked after multiple failed login attempts. "
            "They have an urgent bill payment due today and cannot access their account."
        ),
        "questions": [
            {
                "text": (
                    "I've been locked out of my online banking account. "
                    "I tried logging in a few times and now it says my account is blocked. What do I do?"
                ),
                "topic": "Account Lockout",
            },
            {
                "text": (
                    "I have a bill payment due today — if it goes past midnight I'll be charged a late fee. "
                    "Can you unlock my account right now over the phone?"
                ),
                "topic": "Urgent Unlock Request",
            },
            {
                "text": "I don't remember which password I used. Can you just reset it for me so I can get back in?",
                "topic": "Password Reset",
            },
            {
                "text": (
                    "What verification do you need from me? "
                    "I can give you my account number, date of birth, whatever you need."
                ),
                "topic": "Identity Verification",
            },
            {
                "text": "Once my account is unlocked, is there anything I should do to make sure this doesn't happen again?",
                "topic": "Account Security Tips",
            },
        ],
    },

    "fixed_deposit_inquiry": {
        "title":         "Fixed Deposit Inquiry",
        "call_type_key": "fixed_deposit_inquiry",
        "topic":         "Maturity & Renewal Options",
        "level":         "Advanced",
        "description": (
            "Customer has a fixed deposit maturing in two weeks and wants to understand "
            "their options — renewal, withdrawal, or reinvestment — and is concerned about interest rates."
        ),
        "questions": [
            {
                "text": (
                    "My fixed deposit is maturing in about two weeks. I got a notification but it wasn't very clear. "
                    "What exactly happens if I don't do anything before the maturity date?"
                ),
                "topic": "Auto-Renewal Policy",
            },
            {
                "text": (
                    "What interest rate will I get if I renew it for another year? "
                    "Because the rate I got two years ago was much better than what I'm seeing now."
                ),
                "topic": "Interest Rate Inquiry",
            },
            {
                "text": "Is there a penalty if I withdraw the full amount at maturity instead of renewing?",
                "topic": "Withdrawal Penalty",
            },
            {
                "text": (
                    "I've heard about some new investment schemes the bank is offering. "
                    "Would it make more sense to move this money into one of those instead of a fixed deposit?"
                ),
                "topic": "Alternative Investment Options",
            },
            {
                "text": (
                    "If I decide to partially withdraw and reinvest the rest, is that possible? "
                    "And how do I instruct the bank on what to do before the maturity date?"
                ),
                "topic": "Partial Withdrawal & Reinvestment",
            },
        ],
    },

    "credit_card_dispute": {
        "title":         "Credit Card Dispute",
        "call_type_key": "credit_card_dispute",
        "topic":         "Duplicate Charge Chargeback",
        "level":         "Intermediate",
        "description": (
            "Customer was charged twice for the same online purchase on their credit card "
            "and the merchant is not responding. They want the bank to intervene."
        ),
        "questions": [
            {
                "text": (
                    "I made an online purchase last week for $95 and I've been charged twice for the exact same transaction. "
                    "I can see both charges on my credit card statement. Can you look into this?"
                ),
                "topic": "Duplicate Charge",
            },
            {
                "text": (
                    "I already contacted the merchant three days ago and they haven't replied at all. "
                    "Can the bank raise a dispute on my behalf?"
                ),
                "topic": "Bank Dispute Filing",
            },
            {
                "text": (
                    "How does the dispute process work? "
                    "Will the $95 be temporarily credited back to my account while the investigation is ongoing?"
                ),
                "topic": "Provisional Credit",
            },
            {
                "text": (
                    "How long does a chargeback investigation usually take? "
                    "I don't want this sitting on my bill and accruing interest."
                ),
                "topic": "Investigation Timeline",
            },
            {
                "text": (
                    "Is there anything I need to submit from my side — "
                    "like screenshots or order confirmation — to support the dispute?"
                ),
                "topic": "Supporting Evidence",
            },
        ],
    },
}

# Default scenario loaded when the app starts
DEFAULT_SCENARIO = "fraudulent_transaction"


# ── Helper functions ──────────────────────────────────────────────────────────

def get_scenario(key: str) -> dict:
    """Return a scenario dict or raise KeyError."""
    if key not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{key}'. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[key]


def get_question_texts(scenario_key: str) -> list[str]:
    """Return plain question strings for a scenario (used by prompts.py)."""
    return [q["text"] for q in get_scenario(scenario_key)["questions"]]


def get_scenarios_by_call_type(call_type_key: str) -> dict[str, dict]:
    """Return all scenarios that match a given call_type_key."""
    return {k: v for k, v in SCENARIOS.items() if v["call_type_key"] == call_type_key}


def get_scenarios_by_level(level: str) -> dict[str, dict]:
    """Return all scenarios at a given training level."""
    return {k: v for k, v in SCENARIOS.items() if v.get("level", "").lower() == level.lower()}


def get_all_topics() -> list[dict]:
    """
    Return a flat list of all unique topics across all scenarios.
    Each entry: {"call_type_key": ..., "call_type_title": ..., "topic": ...}
    """
    seen = set()
    topics = []
    for scenario in SCENARIOS.values():
        for q in scenario["questions"]:
            key = (scenario["call_type_key"], q["topic"])
            if key not in seen:
                seen.add(key)
                topics.append({
                    "call_type_key":   scenario["call_type_key"],
                    "call_type_title": scenario["title"],
                    "topic":           q["topic"],
                })
    return topics


def scenario_summary() -> list[dict]:
    """
    Return a lightweight summary list of all scenarios — used by API endpoints.
    """
    return [
        {
            "key":           k,
            "title":         v["title"],
            "call_type_key": v["call_type_key"],
            "topic":         v["topic"],
            "level":         v.get("level", ""),
            "description":   v["description"],
            "question_count": len(v["questions"]),
        }
        for k, v in SCENARIOS.items()
    ]
