# clawdbot

An AI-powered Upwork proposal writer that runs a multi-agent pipeline to produce highly specific, human-sounding proposals — using a mix of OpenAI and Google Gemini models.

---

## How It Works

The pipeline has four sequential steps:

```
Job Post
   │
   ▼
[Step 0] PreAnalysisAgent      — extracts pain point, keywords, urgency, proof points
   │
   ▼
[Step 1] 4 Writer Agents       — draft proposals in parallel (Concise, Detailed, Insight, Data)
   │
   ▼
[Step 2] 2 Evaluator Agents    — score each proposal; ConversionEvaluator is weighted 2x
   │
   ▼
[Step 3] HumanizerAgent        — polishes the winning proposal to sound like a real person wrote it
   │
   ▼
Final Proposal
```

Each proposal follows a strict 3-beat structure: **Hook → Solve It → Reply Question**.

---

## Agent Architecture & Model Choices

Models are deliberately mixed between OpenAI GPT-5.2 and Google Gemini 3. Each agent runs on the model best suited to its specific task.

### Writers

| Agent | Model | Why This Model |
|---|---|---|
| `ConciseWriter` | `gpt-5.2` | Strict 120–150 word hard cap requires GPT-5.2's best-in-class instruction-following and constraint adherence |
| `DetailedWriter` | `gemini-3-pro-preview` | Gemini 3 Pro excels at structured, multi-point writing and naturally mirrors the client's language |
| `InsightWriter` | `gpt-5.2` | Requires nuanced peer-level technical tone and domain empathy — GPT-5.2's strongest writing capability |
| `DataWriter` | `gemini-3-pro-preview` | Gemini 3 Pro is strong at numbers-driven, cause-and-effect structured writing |

### Evaluators

| Agent | Model | Why This Model |
|---|---|---|
| `FormulaEvaluator` (1x weight) | `gemini-3-flash-preview` | Pure rubric/checklist scoring — Gemini 3 Flash is fast, accurate, and cost-efficient for structured eval |
| `ConversionEvaluator` (2x weight) | `gpt-5-mini` | The most critical evaluator (2x weighted); GPT-5 Mini brings GPT-5-class judgment at lower cost |

> **Weighting logic:** `ConversionEvaluator` simulates a busy Upwork client deciding whether to reply. It counts 2x because conversion likelihood is more predictive of real-world success than structural formula adherence.

### Support Agents

| Agent | Model | Why This Model |
|---|---|---|
| `PreAnalysisAgent` | `gemini-3-flash-preview` | Fast structured extraction (pain point, keywords, proof points) — Flash handles this in under 2 seconds |
| `HumanizerAgent` | `gpt-5.2` | Final pass on the winning proposal; output quality here directly determines what the user sends — needs the best writing model |

---

## Model Reference (as of February 2026)

### OpenAI

| Model | Best For |
|---|---|
| `gpt-5.2` | Best overall — coding, writing, agentic tasks, instruction-following |
| `gpt-5-mini` | Fast, cost-efficient GPT-5-class model for well-defined tasks |
| `gpt-5-nano` | Fastest and cheapest GPT-5 variant |
| `gpt-4.1` | Previous best non-reasoning model (still solid) |

> `gpt-4.1` and `gpt-4.1-mini` have been replaced in this project by `gpt-5.2` and `gpt-5-mini`.

### Google Gemini

| Model | Best For |
|---|---|
| `gemini-3-pro-preview` | Highest intelligence — multimodal reasoning, complex writing, agentic tasks |
| `gemini-3-flash-preview` | Balanced speed + frontier-class performance; launched Dec 2025 |

> **`gemini-2.0-flash` is deprecated and will be shut down March 31, 2026.** All Gemini agents have been migrated to Gemini 3.

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Install & Run

```bash
uv sync
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_ai_studio_key
```

> Google API key: get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Enable billing to avoid free-tier quota limits.

### Launch the UI

```bash
uv run python app.py
```

Opens a Gradio chat interface. Paste any Upwork job post and get a proposal back.

---

## Project Structure

```
clawdbot/
├── app.py              # Gradio UI — entry point
├── core/
│   ├── agent.py        # Main multi-model pipeline (GPT-5.2 + Gemini 3)
│   ├── gemmini.py      # Gemini 3 Flash and Pro model instances (used by agent.py)
│   └── prompts.py      # All system prompts, writer styles, evaluator instructions
├── pyproject.toml
└── .env                # API keys (not committed)
```

---

## Proposal Structure

Every proposal written by clawdbot follows a strict 3-beat format:

1. **Hook** — Names the client's exact problem in the first sentence. No greeting. No self-intro.
2. **Solve It** — Specific approach to this exact job. Mirrors client's language. One real proof point (a number, not a job title). Bullets for 3+ items, each starting with "I'll" or "I'd".
3. **Reply Question** — Ends with one specific, low-friction question that invites a one-sentence reply. No fake availability windows. No boilerplate closers.
