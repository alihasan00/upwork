from .prompts import (
    INSTRUCTIONS,
    CONCISE_WRITER_STYLE,
    DETAILED_WRITER_STYLE,
    INSIGHT_WRITER_STYLE,
    DATA_WRITER_STYLE,
    FORMULA_EVALUATOR_INSTRUCTIONS,
    CONVERSION_EVALUATOR_INSTRUCTIONS,
    PRE_ANALYSIS_INSTRUCTIONS,
    HUMANIZER_INSTRUCTIONS,
    PROFILE,
)
import asyncio
import logging
import time
from pydantic import BaseModel
from agents import Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv

set_tracing_disabled(True)
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clawdbot")


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------


class Proposal(BaseModel):
    proposal: str


class EvaluationScore(BaseModel):
    agent_name: str
    score: int  # 1–10
    feedback: str


class ProposalEvaluation(BaseModel):
    scores: list[EvaluationScore]
    winner: str
    reasoning: str


# ---------------------------------------------------------------------------
# Evaluator weights — ConversionEvaluator counts 2x
# ---------------------------------------------------------------------------

EVALUATOR_WEIGHTS: dict[str, float] = {
    "FormulaEvaluator": 1.0,
    "ConversionEvaluator": 2.0,
}

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

concise_writer = Agent(
    name="ConciseWriter",
    model="gpt-4.1",
    instructions=INSTRUCTIONS + CONCISE_WRITER_STYLE,
    output_type=Proposal,
)

detailed_writer = Agent(
    name="DetailedWriter",
    model="gpt-4.1",
    instructions=INSTRUCTIONS + DETAILED_WRITER_STYLE,
    output_type=Proposal,
)

insight_writer = Agent(
    name="InsightWriter",
    model="gpt-4.1",
    instructions=INSTRUCTIONS + INSIGHT_WRITER_STYLE,
    output_type=Proposal,
)

data_writer = Agent(
    name="DataWriter",
    model="gpt-4.1",
    instructions=INSTRUCTIONS + DATA_WRITER_STYLE,
    output_type=Proposal,
)

formula_evaluator = Agent(
    name="FormulaEvaluator",
    model="gpt-4.1-mini",
    instructions=FORMULA_EVALUATOR_INSTRUCTIONS,
    output_type=ProposalEvaluation,
)

conversion_evaluator = Agent(
    name="ConversionEvaluator",
    model="gpt-4.1-mini",
    instructions=CONVERSION_EVALUATOR_INSTRUCTIONS,
    output_type=ProposalEvaluation,
)

pre_analysis_agent = Agent(
    name="PreAnalysisAgent",
    model="gpt-4.1-mini",
    instructions=PRE_ANALYSIS_INSTRUCTIONS,
)

humanizer_agent = Agent(
    name="HumanizerAgent",
    model="gpt-4.1",
    instructions=HUMANIZER_INSTRUCTIONS,
    output_type=Proposal,
)

WRITERS = [concise_writer, detailed_writer, insight_writer, data_writer]
WRITER_STYLES = [
    CONCISE_WRITER_STYLE,
    DETAILED_WRITER_STYLE,
    INSIGHT_WRITER_STYLE,
    DATA_WRITER_STYLE,
]
EVALUATORS = [formula_evaluator, conversion_evaluator]

# Word limits per writer style — used in quality gates
WRITER_WORD_LIMITS: dict[str, int] = {
    "ConciseWriter": 150,
    "DetailedWriter": 240,
    "InsightWriter": 200,
    "DataWriter": 220,
}

_BANNED_PHRASES = [
    "ensure",
    "leverage",
    "utilize",
    "seamless",
    "robust",
    "cutting-edge",
    "synergy",
    "streamline",
    "delve",
    "look no further",
    "i would love to",
    "feel free to",
    "don't hesitate",
    "i'm the perfect fit",
    "with that said",
    "i am confident",
    "proven track record",
    "best practices",
    "end-to-end solution",
    "tailored solution",
    "tailored approach",
    "i came across your posting",
    "i'm reaching out",
    "i look forward to hearing",
    "as per your",
    "touch base",
]


# ---------------------------------------------------------------------------
# Quality gate — logs warnings, does not block
# ---------------------------------------------------------------------------


def _check_quality_gates(proposals: dict[str, str]) -> None:
    for name, text in proposals.items():
        word_count = len(text.split())
        limit = WRITER_WORD_LIMITS.get(name, 250)
        if word_count > limit:
            log.warning(
                "  QUALITY GATE: %s is %d words (limit %d)",
                name,
                word_count,
                limit,
            )
        found = [p for p in _BANNED_PHRASES if p in text.lower()]
        if found:
            log.warning(
                "  QUALITY GATE: %s contains banned phrases: %s",
                name,
                found,
            )


# ---------------------------------------------------------------------------
# Pipeline — returns the winning proposal as a plain string
# ---------------------------------------------------------------------------


async def run_pipeline(job_post: str, custom_instructions: str = "") -> str:
    if not job_post.strip():
        raise ValueError("job_post cannot be empty")

    pipeline_start = time.perf_counter()
    log.info("Pipeline started — job post: %d chars", len(job_post.strip()))

    base = custom_instructions.strip() or INSTRUCTIONS

    # --- Step 0: Pre-analysis ---
    log.info("Step 0 — pre-analysis: extracting job signals and proof points")
    t0 = time.perf_counter()

    pre_result = await Runner.run(
        pre_analysis_agent,
        input=f"Job post:\n{job_post}\n\nFreelancer profile:\n{PROFILE}",
    )
    signals_text: str = pre_result.final_output
    log.info("Step 0 done — %.1fs\n%s", time.perf_counter() - t0, signals_text)

    writer_context = (
        base
        + "\n\n---\n\nJOB SIGNALS (extracted for this specific job post):\n"
        + signals_text
        + "\n\nIMPORTANT: Use only the PROOF POINTS listed above when adding credentials. "
        + "Do not pull from the full profile.\n"
    )

    writers = [
        Agent(
            name=w.name,
            model=w.model,
            instructions=writer_context + style,
            output_type=Proposal,
        )
        for w, style in zip(WRITERS, WRITER_STYLES)
    ]

    # --- Step 1: Writers ---
    log.info(
        "Step 1 — running %d writer agents in parallel: %s",
        len(writers),
        [w.name for w in writers],
    )
    t0 = time.perf_counter()

    write_results = await asyncio.gather(
        *[
            Runner.run(
                w, input=f"Write a proposal for this Upwork job post:\n\n{job_post}"
            )
            for w in writers
        ]
    )

    log.info("Step 1 done — %.1fs", time.perf_counter() - t0)

    proposals = {
        writers[i].name: write_results[i].final_output.proposal
        for i in range(len(writers))
    }
    for name, text in proposals.items():
        log.info("  %-20s %d words", name, len(text.split()))

    # --- Quality gate ---
    _check_quality_gates(proposals)

    # --- Step 2: Evaluators ---
    eval_prompt = f"Original job post:\n{job_post}\n\n"
    for name, text in proposals.items():
        eval_prompt += f"--- {name} ---\n{text}\n\n"

    log.info(
        "Step 2 — running %d evaluator agents in parallel: %s",
        len(EVALUATORS),
        [e.name for e in EVALUATORS],
    )
    t0 = time.perf_counter()

    eval_results = await asyncio.gather(
        *[Runner.run(e, input=eval_prompt) for e in EVALUATORS]
    )

    log.info("Step 2 done — %.1fs", time.perf_counter() - t0)

    for result, evaluator in zip(eval_results, EVALUATORS):
        ev: ProposalEvaluation = result.final_output
        log.info("  %s → winner: %s", evaluator.name, ev.winner)
        for s in ev.scores:
            log.info("    %-20s %d/10", s.agent_name, s.score)

    # --- Combine scores (ConversionEvaluator weighted 2x) ---
    combined: dict[str, float] = {name: 0.0 for name in proposals}
    total_weight = sum(EVALUATOR_WEIGHTS.get(e.name, 1.0) for e in EVALUATORS)

    for result, evaluator in zip(eval_results, EVALUATORS):
        weight = EVALUATOR_WEIGHTS.get(evaluator.name, 1.0)
        for s in result.final_output.scores:
            if s.agent_name in combined:
                combined[s.agent_name] += s.score * weight

    winner = max(combined, key=lambda k: combined[k])
    avg = combined[winner] / total_weight
    log.info("  Combined scores (weighted): %s", combined)

    # --- Step 3: Humanizer pass on winner ---
    log.info("Step 3 — humanizer pass on winner: %s", winner)
    t0 = time.perf_counter()

    humanized_result = await Runner.run(
        humanizer_agent,
        input=f"Original job post:\n{job_post}\n\nProposal to humanize:\n{proposals[winner]}",
    )
    final_proposal = humanized_result.final_output.proposal
    log.info("Step 3 done — %.1fs", time.perf_counter() - t0)

    log.info(
        "Pipeline complete — winner: %s (%.1f/10) — total: %.1fs",
        winner,
        avg,
        time.perf_counter() - pipeline_start,
    )

    return f"**Winner: {winner} ({avg:.1f}/10)**\n\n{final_proposal}"
