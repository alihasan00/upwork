from .prompts import (
    INSTRUCTIONS,
    CONCISE_WRITER_STYLE,
    DETAILED_WRITER_STYLE,
    INSIGHT_WRITER_STYLE,
    EVALUATOR_ORCHESTRATOR_INSTRUCTIONS,
    PRE_ANALYSIS_INSTRUCTIONS,
    HUMANIZER_INSTRUCTIONS,
    REVISION_INSTRUCTIONS,
    PROFILE,
)
from .gemmini import gemini_flash_model, gemini_pro_model
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


# ---------------------------------------------------------------------------
# Static agents (no per-job context needed)
# ---------------------------------------------------------------------------

pre_analysis_agent = Agent(
    name="PreAnalysisAgent",
    model=gemini_flash_model,  # fast structured signal extraction
    instructions=PRE_ANALYSIS_INSTRUCTIONS,
)

# Terminal agent — no handoffs
humanizer_agent = Agent(
    name="HumanizerAgent",
    model="gpt-5.2",
    instructions=HUMANIZER_INSTRUCTIONS,
    output_type=Proposal,
)

# Revises the winning proposal, then hands off to humanizer
revision_agent = Agent(
    name="RevisionAgent",
    model="gpt-5.2",
    instructions=REVISION_INSTRUCTIONS,
    output_type=Proposal,
    handoffs=[humanizer_agent],
)

# ---------------------------------------------------------------------------
# Writer templates — cloned inside run_pipeline() with injected context
# ---------------------------------------------------------------------------

WRITERS = [
    Agent(
        name="ConciseWriter",
        model="gpt-5.2",
        instructions=INSTRUCTIONS + CONCISE_WRITER_STYLE,
        output_type=Proposal,
    ),
    Agent(
        name="DetailedWriter",
        model=gemini_pro_model,
        instructions=INSTRUCTIONS + DETAILED_WRITER_STYLE,
        output_type=Proposal,
    ),
    Agent(
        name="InsightWriter",
        model="gpt-5.2",
        instructions=INSTRUCTIONS + INSIGHT_WRITER_STYLE,
        output_type=Proposal,
    ),
]

WRITER_STYLES = [
    CONCISE_WRITER_STYLE,
    DETAILED_WRITER_STYLE,
    INSIGHT_WRITER_STYLE,
]

WRITER_TOOL_DESCRIPTIONS = [
    "Write a concise 120–150 word Upwork proposal with a blunt problem-first opener.",
    "Write a detailed 200–240 word Upwork proposal mirroring the client's language.",
    "Write an insight-driven 160–200 word Upwork proposal with a competence-based opener.",
]


# ---------------------------------------------------------------------------
# Pipeline
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

    # Build writer agents with job-specific context injected into their instructions
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

    # --- Steps 1–3: EvaluatorOrchestrator drives everything from here ---
    # It calls writers as tools, scores proposals, retries if score < 7,
    # then hands off → RevisionAgent → HumanizerAgent
    log.info(
        "Step 1–3 — evaluator orchestrator: writers=%s, retry threshold=7/10",
        [w.name for w in writers],
    )
    t0 = time.perf_counter()

    evaluator_orchestrator = Agent(
        name="EvaluatorOrchestrator",
        model="gpt-5.2",  # evaluates, decides retries, constructs handoff
        instructions=EVALUATOR_ORCHESTRATOR_INSTRUCTIONS,
        tools=[
            writers[i].as_tool(
                tool_name=writers[i].name.lower(),
                tool_description=WRITER_TOOL_DESCRIPTIONS[i],
            )
            for i in range(len(writers))
        ],
        handoffs=[revision_agent],
    )

    result = await Runner.run(
        evaluator_orchestrator,
        input=(
            f"Job post:\n{job_post}\n\n"
            f"Job signals:\n{signals_text}"
        ),
    )

    log.info(
        "Pipeline complete — total: %.1fs",
        time.perf_counter() - pipeline_start,
    )

    final: Proposal = result.final_output
    return final.proposal
