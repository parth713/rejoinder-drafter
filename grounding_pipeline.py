"""Loan Arbitration Graph Grounding pipeline.

Implements the `loan-arbitration-graph-grounding` skill on top of Ajay's
Rejoinder Drafter:

  Step 0 + 1  -> extract_graph()         (one LLM pass: posture + node extraction)
  Step 4 + 5  -> build_context_packs()   (scoped, suppressed, per-issue packs)
  Step 7      -> stream_grounded_rejoinder()
  Step 8      -> coverage_report()

There is no Neo4j: the case graph is small JSON, so "scoped retrieval" is done in
plain Python. The point is faithful to the skill — the LLM that drafts the
Rejoinder only ever sees the minimum subgraph that rebuts each contention, with
every fact carrying provenance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from app import ExtractedDocument, SYSTEM_PROMPT, build_drafting_input

GROUNDING_DIR = Path(__file__).resolve().parent / "grounding"
SKILL_PATH = GROUNDING_DIR / "SKILL.md"
EXAMPLE_GRAPH_PATH = GROUNDING_DIR / "tata_agarwal_case_graph.json"
SKELETON_EX_PARTE_PATH = GROUNDING_DIR / "skeleton_ex_parte_example.json"

# Marker used by the extractor for values that must never reach the draft.
SENSITIVE_MARKER = "[SENSITIVE"
SENSITIVE_PLACEHOLDER = "[SENSITIVE — withheld]"


# --------------------------------------------------------------------------- #
# Loading the shipped skill assets
# --------------------------------------------------------------------------- #
def load_skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def load_example_graph() -> dict:
    """The Tata-Agarwal graph — a real, fact-rich MERITS-branch worked instance.

    This file holds real client PII and is git-ignored, so a fresh clone will not
    have it. Supply it locally at grounding/tata_agarwal_case_graph.json.
    """
    if not EXAMPLE_GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EXAMPLE_GRAPH_PATH.name}. It contains client PII and is not "
            "committed; place the real graph at grounding/tata_agarwal_case_graph.json "
            "to enable graph extraction and the bundled-example mode."
        )
    return json.loads(EXAMPLE_GRAPH_PATH.read_text(encoding="utf-8"))


def load_skeleton_ex_parte() -> dict:
    """A placeholder-only PROCEDURAL-GATE skeleton (no real facts) for shape."""
    return json.loads(SKELETON_EX_PARTE_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Step 0 + 1 — Extract the grounded knowledge graph
# --------------------------------------------------------------------------- #
EXTRACTION_INSTRUCTIONS = """You are a litigation knowledge-graph extractor for Indian loan-arbitration matters.

Apply the skill below to the supplied case documents and output ONE knowledge graph as strict JSON.

Hard rules:
- Run Step 0 (posture detection) first. Set graph_metadata.filing_formality, graph_metadata.proceeding_mode, and graph_metadata.procedural_posture. The branch follows proceeding_mode.
- Run Step 1: create one node per ontology node actually present in the record. Populate ONLY attributes the documents support; use null or "NOT IN RECORD" otherwise. NEVER invent facts, dates, amounts, clauses, exhibit numbers, or parties.
- Every node MUST carry a "provenance" array of "<DOC> p.<n>" strings drawn from the document tags/pages you were given. No fact without provenance.
- Mark sensitive values (bank account numbers, IFSC, card numbers, government IDs, phone, email) as the string "[SENSITIVE — withheld from generative context]". Do not output their real values.
- Suppressed/empty nodes (no co-borrower, no guarantor, unsecured, etc.) must still appear with "suppressed": true and a one-line note.
- Regenerate case-specific discrepancy flags (Step 6) and attach them to the affected nodes and to graph_metadata.active_flags.
- Build issues_to_relevance_map[]: one entry per discrete respondent contention, each mapping to the MINIMUM rebuttal_nodes (id + props_used attribute names + provenance) that answer it, with a posture_tag.
- Add a coverage_check object (Step 8).

OUTPUT SCHEMA — return a single JSON object with EXACTLY these top-level keys (the examples demonstrate the shape; the contract below governs):
{
  "graph_metadata": {
    "matter": str, "case_reference": str, "loan_account": str|null,
    "filing_formality": "formal_SOD" | "informal_representations" | "no_filing",
    "proceeding_mode": "inter_partes" | "ex_parte",
    "procedural_posture": str (MERITS or PROCEDURAL-GATE, with a one-paragraph justification),
    "active_flags": [str, ...]   // regenerate per case; do not reuse another case's flags
  },
  "nodes": [
    {
      "id": str, "label": str (an ontology node type),
      "props": { <only attributes the documents support>, "provenance": ["<DOC> p.<n>", ...] },
      "suppressed": bool (optional; true for empty/"not applicable" nodes — these still get a props.note),
      "_extension": true (REQUIRED on InterimMeasure/ProceduralOrder/AttachedAsset/ThirdParty),
      "flags": [str, ...] (optional)
    }
  ],
  "relationships": [ { "from": <node id>, "type": <relationship type>, "to": <node id> } ],
  "issues_to_relevance_map": [
    {
      "issue": str (respondent plea + para no.), "claimant_thrust": str,
      "rebuttal_nodes": [ { "id": <node id>, "props_used": [<attribute names>], "provenance": ["<DOC> p.<n>", ...] } ],
      "flags": [str, ...], "posture_tag": "merits" | "gate_threshold" | "alternative_without_prejudice"
    }
  ],
  "coverage_check": { "dominant_battlegrounds": [str, ...], "each_backed_by_provenance": bool, "empties_walled_off": [<node id>, ...] }
}
Conventions: provenance strings are "<DOC> p.<n>" drawn from the document tags/pages supplied. Absent value -> null or "NOT IN RECORD". Sensitive value -> "[SENSITIVE — withheld from generative context]". In the PROCEDURAL-GATE branch the gate issue's posture_tag is "gate_threshold" and leads; merits issues are "alternative_without_prejudice".
Output JSON only. No prose, no markdown fences."""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def extract_graph(
    client: OpenAI,
    model: str,
    documents: list[ExtractedDocument],
    max_output_tokens: int = 16_000,
) -> dict:
    """Step 0 + Step 1: one LLM call that returns the grounded case graph."""
    skill_text = load_skill_text()
    merits_example = json.dumps(load_example_graph(), ensure_ascii=False, indent=2)
    gate_skeleton = json.dumps(load_skeleton_ex_parte(), ensure_ascii=False, indent=2)
    documents_block = build_drafting_input(documents, "Extract the graph only; do not draft.")

    extraction_input = f"""<skill>
{skill_text}
</skill>

<merits_branch_example role="REAL worked instance, MERITS branch. Shows the JSON shape, provenance/suppression conventions, and judgement. Match the shape; do NOT copy its facts.">
{merits_example}
</merits_branch_example>

<procedural_gate_skeleton role="STRUCTURAL skeleton, PROCEDURAL-GATE/ex-parte branch. All values are PLACEHOLDERS, not facts. Shows gate-first ordering, the four ontology extensions, and an active co-borrower. Use ONLY if the case is ex-parte; never copy any placeholder value.">
{gate_skeleton}
</procedural_gate_skeleton>

Pick the branch from Step 0 posture detection on the documents below — do not assume the merits branch just because its example is richer.

{documents_block}

Now produce the grounded knowledge-graph JSON for the case documents above, following the OUTPUT SCHEMA."""

    response = client.responses.create(
        model=model,
        instructions=EXTRACTION_INSTRUCTIONS,
        input=extraction_input,
        max_output_tokens=max_output_tokens,
    )
    raw = _strip_json_fences(response.output_text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The graph extractor did not return valid JSON: {exc}.\n\nRaw start:\n{raw[:800]}"
        ) from exc


# --------------------------------------------------------------------------- #
# Step 4 + 5 — Scoped retrieval + suppression -> per-issue context packs
# --------------------------------------------------------------------------- #
def _is_suppressed(node: dict) -> bool:
    if node.get("suppressed"):
        return True
    return node.get("props", {}).get("present") is False


def _scope_props(node: dict, props_used) -> dict:
    """Attribute-level scoping (rule 2) + sensitive withholding (rule 9)."""
    props = node.get("props", {})
    if not props_used:
        selected = dict(props)
    else:
        selected = {key: props.get(key) for key in props_used if key in props}

    def scrub(value):
        if isinstance(value, str) and SENSITIVE_MARKER in value:
            return SENSITIVE_PLACEHOLDER
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    return {key: scrub(value) for key, value in selected.items()}


def build_context_packs(graph: dict) -> list[dict]:
    """Steps 4-5: resolve each issue to its minimum, scoped, provenance-tagged pack."""
    nodes_by_id = {node.get("id"): node for node in graph.get("nodes", [])}
    posture = graph.get("graph_metadata", {}).get("procedural_posture", "") or ""
    is_gate = "gate" in posture.lower() or "ex-parte" in posture.lower() or "ex_parte" in posture.lower()

    packs: list[dict] = []
    for issue in graph.get("issues_to_relevance_map", []):
        rebuttal_nodes = []
        for ref in issue.get("rebuttal_nodes", []):
            node = nodes_by_id.get(ref.get("id"))
            if node is None or _is_suppressed(node):
                continue  # rule 1: empty/suppressed nodes are walled off from generation
            rebuttal_nodes.append(
                {
                    "id": ref.get("id"),
                    "props_used": _scope_props(node, ref.get("props_used")),
                    "provenance": ref.get("provenance") or node.get("provenance", []),
                }
            )

        posture_tag = issue.get("posture_tag", "merits")
        # rule 7: in the procedural-gate branch, merits content is alternative/without prejudice.
        if is_gate and posture_tag == "merits":
            posture_tag = "alternative_without_prejudice"

        packs.append(
            {
                "issue": issue.get("issue", ""),
                "claimant_thrust": issue.get("claimant_thrust", ""),
                "rebuttal_nodes": rebuttal_nodes,
                "flags": issue.get("flags", []),
                "posture_tag": posture_tag,
                "instruction": (
                    "Draft only from props_used. Assert nothing outside this pack. "
                    "Cite provenance inline. Do not emit sensitive identifiers."
                ),
            }
        )
    return packs


def suppressed_facts(graph: dict) -> list[str]:
    """Rule 1: empty nodes passed only as one-line 'not applicable' facts."""
    lines = []
    for node in graph.get("nodes", []):
        if _is_suppressed(node):
            note = node.get("props", {}).get("note", "not applicable")
            lines.append(f"{node.get('label', 'Node')} ({node.get('id')}): {note}")
    return lines


# --------------------------------------------------------------------------- #
# Step 7 — Draft the Rejoinder from the packs
# --------------------------------------------------------------------------- #
GROUNDED_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

GROUNDING OVERLAY (these rules sit on top of the drafting rules above and win on conflict):
G1. Draft strictly from the per-issue context packs supplied below. Every factual assertion must come from a pack's props_used. Do not introduce any fact, party, asset, guarantor, amount, date, or authority that is not in a pack.
G2. Cite provenance inline for each material assertion, in the form (SOA p.1), exactly as carried in the pack. No fact without provenance.
G3. Honour each pack's posture_tag. Tags of "gate_threshold" lead the document; tags of "alternative_without_prejudice" are pleaded only in the alternative and expressly without prejudice to the threshold objection — never concede the gate.
G4. If a flag indicates the respondent's reply answers a different lis (e.g. reply_addresses_different_lis), do NOT treat the claim as joined or admitted; plead that the averment stands unrebutted and the respondent is put to strict proof.
G5. Never emit sensitive identifiers. Where a value is withheld, refer to it descriptively (e.g. "the registered repayment account").
G6. Surface each pack's flags as drafting caution where relevant (e.g. correct a mis-cited clause number rather than repeating it)."""
)


def _serialize_packs(packs: list[dict], suppressed: list[str], posture: dict) -> str:
    return json.dumps(
        {
            "posture": posture,
            "suppressed_not_applicable_facts": suppressed,
            "issue_packs": packs,
        },
        ensure_ascii=False,
        indent=2,
    )


def stream_grounded_rejoinder(
    client: OpenAI,
    model: str,
    packs: list[dict],
    suppressed: list[str],
    posture: dict,
    drafting_instructions: str,
    max_output_tokens: int,
) -> Iterable[str]:
    instructions = drafting_instructions.strip() or "No additional instructions supplied."
    grounding_payload = _serialize_packs(packs, suppressed, posture)
    drafting_input = f"""Draft the Claimant's complete Rejoinder using ONLY the grounded context packs below.

ADDITIONAL COUNSEL INSTRUCTIONS
{instructions}

GROUNDED CONTEXT (posture, suppressed facts, and one pack per respondent contention)
{grounding_payload}

Address every issue pack. Lead with any gate_threshold packs; plead alternative_without_prejudice packs only in the alternative. Cite provenance inline. Output only the final Markdown Rejoinder."""

    with client.responses.stream(
        model=model,
        instructions=GROUNDED_SYSTEM_PROMPT,
        input=drafting_input,
        max_output_tokens=max_output_tokens,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta


# --------------------------------------------------------------------------- #
# Step 8 — Coverage check
# --------------------------------------------------------------------------- #
def coverage_report(graph: dict, packs: list[dict]) -> dict:
    """Step 8: every plea maps to >=1 rebuttal node; surface the graph's own check."""
    unbacked = [pack["issue"] for pack in packs if not pack["rebuttal_nodes"]]
    return {
        "issues_total": len(packs),
        "issues_backed": len(packs) - len(unbacked),
        "issues_unbacked": unbacked,
        "graph_coverage_check": graph.get("coverage_check", {}),
        "active_flags": graph.get("graph_metadata", {}).get("active_flags", []),
    }


def posture_summary(graph: dict) -> dict:
    meta = graph.get("graph_metadata", {})
    return {
        "matter": meta.get("matter", ""),
        "filing_formality": meta.get("filing_formality", ""),
        "proceeding_mode": meta.get("proceeding_mode", ""),
        "procedural_posture": meta.get("procedural_posture", ""),
    }
