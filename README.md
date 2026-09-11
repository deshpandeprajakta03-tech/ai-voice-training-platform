# AI Voice Training Platform

An AI-based voice training application that simulates realistic phone conversations between customer-support representatives and AI-generated customers.

The system is built on a base voice simulator originally developed by my professor. I extended it by adding the database layer, GPT-4.1 transcript extraction pipeline, REST API endpoints, and the full management dashboard.

---

## Demo

[![Watch the demo](https://img.youtube.com/vi/F5r6Q6-BSms/maxresdefault.jpg)](https://youtu.be/F5r6Q6-BSms)

---

## What it does

- The **AI bot plays the customer** and asks questions based on real banking scenarios
- The **trainee plays the bank agent** and responds via microphone
- After the call, **GPT-4 automatically scores** the representative on 5 performance parameters
- Managers can **add trainees, assign them to scenarios, and track their progress** through the dashboard

---

## Features

### Voice Simulator
- Real-time voice conversation powered by Azure OpenAI Realtime API
- Bot asks questions one at a time, waits for the agent to respond
- Supports 5 banking scenarios — Fraudulent Transaction, Loan Application, Account Locked, Fixed Deposit, Credit Card Dispute

### GPT-4.1 Transcript Extraction
- Paste a redacted historical call transcript
- GPT-4.1 automatically extracts valid customer questions
- Excludes greetings, authentication exchanges, and call-closure lines
- Organises extracted questions by call type and topic
- Saves directly to the database with one click

### Training Levels
- Three difficulty tiers — Beginner, Intermediate, Advanced
- Full create, update, delete management

### Call Types & Questions
- Five predefined call categories matching real banking scenarios
- Questions tagged with call type, topic, and difficulty level
- Filter questions by call type or source (manual / extracted)

### Representatives
- Register and manage customer-support trainees
- Track registration date and assignment history

### Assignments
- Assign representatives to specific training levels and call types
- Track status — Pending, In Progress, Completed
- Filter by representative or status

### Feedback History
- All scored feedback reports stored in the database
- Browse by representative or scenario
- Each report shows scores for 5 parameters with comments

---

## Performance Parameters (Feedback Report)

| Parameter | What it measures |
|---|---|
| Greeting & Opening | Professional introduction |
| Tone & Empathy | Calmness and empathy toward the customer |
| Problem Solving | Understanding and resolving the issue |
| Communication Clarity | Clear, jargon-free responses |
| Closing & Wrap-up | Summary and polite call closure |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Voice AI | Azure OpenAI Realtime API |
| Question Extraction | GPT-4.1 |
| Database | SQLAlchemy, SQLite |
| Frontend | HTML, CSS, JavaScript |
| Real-time Audio | WebSockets, AudioWorklet |

---

## Project Structure

```
├── main.py                  # FastAPI app with all API endpoints
├── models.py                # SQLAlchemy database models
├── scenarios.py             # Training scenarios with call type and topic tagging
├── transcript_extractor.py  # GPT-4.1 powered question extraction
├── prompts.py               # System prompt for the AI customer bot
├── azure_bridge.py          # Azure OpenAI Realtime API WebSocket bridge
├── feedback.py              # Post-call feedback report generation
├── index.html               # Frontend — 6-tab management UI
├── requirements.txt         # Python dependencies
└── .env                     # API keys (not committed)
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/scenarios` | List all training scenarios |
| WS | `/ws/audio` | Real-time voice WebSocket |
| POST | `/feedback` | Generate feedback report |
| GET/POST | `/api/levels` | Manage training levels |
| GET/POST | `/api/call-types` | Manage call categories |
| GET/POST | `/api/questions` | Manage training questions |
| GET/POST | `/api/representatives` | Manage trainees |
| GET/POST | `/api/assignments` | Manage assignments |
| GET | `/api/feedback-results` | View stored feedback |
| POST | `/api/extract` | Extract questions via GPT-4.1 |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment variables
Create a `.env` file in the project root:
```
AZURE_OPENAI_CHAT_ENDPOINT=your_endpoint
AZURE_OPENAI_CHAT_KEY=your_key
AZURE_OPENAI_CHAT_DEPLOYMENT=your_deployment
AZURE_OPENAI_REALTIME_ENDPOINT=your_realtime_endpoint
AZURE_OPENAI_REALTIME_KEY=your_realtime_key
AZURE_OPENAI_REALTIME_DEPLOYMENT=your_realtime_deployment
```

### 3. Run the app
```bash
python main.py
```

### 4. Open in browser
```
http://127.0.0.1:8000
```

---

## What I Built

The original project provided by my professor included the core voice simulator — the WebSocket audio bridge and the basic bot conversation flow.

I extended the project by building:
- **Database layer** — 6 SQLAlchemy models for full data persistence
- **GPT-4.1 extraction module** — automatic question generation from transcripts
- **Scenario organisation** — call type, topic, and difficulty level tagging
- **REST API** — 10 endpoints with full CRUD operations
- **Management dashboard** — 6-tab frontend for complete platform management
