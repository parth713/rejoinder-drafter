"""Side-by-side comparison: original raw-dump Rejoinder vs graph-grounded Rejoinder.

Left column  = Ajay's original pipeline (raw document dump -> LLM).
Right column = the loan-arbitration-graph-grounding skill applied
               (extract graph -> scoped per-issue packs -> LLM).

Both columns use the SAME OpenAI model so the only variable is grounding.
"""

from __future__ import annotations

import json
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from app import (
    SUPPORTED_TYPES,
    build_drafting_input,
    stream_rejoinder,
    _extract_all,
)
from grounding_pipeline import (
    build_context_packs,
    coverage_report,
    extract_graph,
    load_example_graph,
    posture_summary,
    stream_grounded_rejoinder,
    suppressed_facts,
)
import neo4j_graph as ng

load_dotenv()


def _run_original(client, model, documents, instructions, max_tokens, placeholder):
    drafting_input = build_drafting_input(documents, instructions)
    chunks: list[str] = []
    for chunk in stream_rejoinder(client, model, drafting_input, max_tokens):
        chunks.append(chunk)
        placeholder.markdown("".join(chunks))
    return "".join(chunks).strip()


def _run_grounded(client, model, documents, instructions, max_tokens, use_example, placeholder):
    if use_example:
        graph = load_example_graph()
    else:
        graph = extract_graph(client, model, documents)

    packs = build_context_packs(graph)
    suppressed = suppressed_facts(graph)
    posture = posture_summary(graph)

    chunks: list[str] = []
    for chunk in stream_grounded_rejoinder(
        client, model, packs, suppressed, posture, instructions, max_tokens
    ):
        chunks.append(chunk)
        placeholder.markdown("".join(chunks))

    return "".join(chunks).strip(), graph, packs, posture


def _render_artifacts(results: dict) -> None:
    graph = results.get("graph") or {}
    st.markdown("**Grounding artifacts**")
    with st.expander("Posture (Step 0)", expanded=False):
        st.json(results.get("posture"))
    with st.expander("Active flags (Step 6)"):
        st.json(graph.get("graph_metadata", {}).get("active_flags", []))
    with st.expander("Per-issue context packs (Steps 4-5, 7)"):
        st.json(results.get("packs"))
    with st.expander("Coverage check (Step 8)"):
        st.json(results.get("coverage"))
    with st.expander("Full extracted graph (Step 1)"):
        st.json(graph)


def _render_columns_static(results: dict) -> None:
    """Re-render the two drafts from session_state (keeps answers intact on reruns)."""
    col_o, col_g = st.columns(2)
    with col_o:
        st.subheader("① Original (raw document dump)")
        if results.get("original_text"):
            st.markdown(results["original_text"])
            st.download_button(
                "Download original (.md)", results["original_text"].encode("utf-8"),
                "claimant_rejoinder.md", "text/markdown",
                use_container_width=True, key="dl_o",
            )
        elif results.get("original_error"):
            st.error(f"Original draft failed: {results['original_error']}")
    with col_g:
        st.subheader("② Graph-grounded (skill applied)")
        if results.get("grounded_text"):
            st.markdown(results["grounded_text"])
            st.download_button(
                "Download grounded (.md)", results["grounded_text"].encode("utf-8"),
                "claimant_rejoinder_grounded.md", "text/markdown",
                use_container_width=True, key="dl_g",
            )
            _render_artifacts(results)
        elif results.get("grounded_error"):
            st.error(f"Grounded draft failed: {results['grounded_error']}")


def _render_graph_section(results: dict) -> None:
    """Full-width interactive graph, read back live from Neo4j."""
    st.divider()
    st.subheader("③ Case graph (live from Neo4j)")
    if not ng.neo4j_configured():
        st.info(
            "Neo4j is not configured — add NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD "
            "and NEO4J_DATABASE to `.env` to auto-load and visualize the graph."
        )
        return
    if results.get("neo4j_error"):
        st.warning(f"Graph could not be loaded into Neo4j: {results['neo4j_error']}")
        return
    sg = results.get("subgraph")
    if not sg:
        return
    show_suppressed = st.toggle(
        "Show suppressed (NA) nodes", value=False, key="show_suppressed_toggle"
    )
    counts = sg.get("counts", {})
    st.caption(
        f"Case `{sg['case_id']}` — {counts.get('nodes', '?')} nodes / "
        f"{counts.get('relationships', '?')} relationships in Neo4j (flushed before load)."
    )
    try:
        import streamlit.components.v1 as components

        html = ng.render_case_html(sg["case_id"], show_suppressed)
        components.html(html, height=680, scrolling=False)
    except Exception as exc:
        st.error(f"Could not render the graph: {exc}")


def main() -> None:
    st.set_page_config(
        page_title="Rejoinder Drafter — Original vs Grounded",
        page_icon="⚖️",
        layout="wide",
    )

    st.title("Rejoinder Drafter — Original vs Graph-Grounded")
    st.caption(
        "Upload the case record once and compare Ajay's original raw-dump draft "
        "against the loan-arbitration-graph-grounding draft, side by side."
    )

    with st.sidebar:
        st.header("Configuration")
        configured_model = os.getenv("OPENAI_MODEL", "").strip()
        if configured_model:
            st.success(f"Model (both sides): {configured_model}")
        else:
            st.error("OPENAI_MODEL is missing from .env")
        max_output_tokens = st.number_input(
            "Maximum output tokens (per draft)",
            min_value=4_000,
            max_value=100_000,
            value=30_000,
            step=1_000,
        )
        use_example = st.checkbox(
            "Use bundled example graph (skip extraction)",
            value=False,
            help=(
                "Grounds the right column on grounding/tata_agarwal_case_graph.json "
                "instead of extracting from the uploads. Lets you test the grounded "
                "path without the source PDFs."
            ),
        )
        st.divider()
        st.warning(
            "AI-generated legal work requires review by qualified counsel. "
            "Uploaded text is sent to the configured model provider."
        )

    with st.form("compare_form"):
        left, right = st.columns(2)
        with left:
            claim_file = st.file_uploader(
                "Statement of Claim *",
                type=SUPPORTED_TYPES,
                help="Primary structure, numbering, and drafting style.",
            )
        with right:
            defence_file = st.file_uploader(
                "Statement of Defence / Reply *",
                type=SUPPORTED_TYPES,
                help="The pleading both drafts must respond to.",
            )
        supporting_files = st.file_uploader(
            "Supporting documents (optional)",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
        )
        drafting_instructions = st.text_area(
            "Additional drafting instructions (optional)",
            height=110,
        )
        submitted = st.form_submit_button(
            "Draft both Rejoinders", type="primary", use_container_width=True
        )

    if submitted:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "").strip()
        if not claim_file or not defence_file:
            st.error("Please upload both the Statement of Claim and the Statement of Defence / Reply.")
        elif not api_key or not model:
            st.error("Set OPENAI_API_KEY and OPENAI_MODEL in .env, then restart the app.")
        else:
            try:
                with st.spinner("Reading the uploaded record…"):
                    documents = _extract_all(claim_file, defence_file, supporting_files or [])
            except Exception as exc:
                st.error(f"Could not read the uploads: {exc}")
                documents = None

            if documents is not None:
                client = OpenAI(api_key=api_key)
                instructions = drafting_instructions
                max_tokens = int(max_output_tokens)
                results: dict = {}

                col_original, col_grounded = st.columns(2)
                with col_original:
                    st.subheader("① Original (raw document dump)")
                    original_placeholder = st.empty()
                with col_grounded:
                    st.subheader("② Graph-grounded (skill applied)")
                    grounded_placeholder = st.empty()

                # --- Original draft (live stream) ---------------------------
                with col_original:
                    try:
                        with st.status("Drafting (original)…", expanded=False) as status:
                            results["original_text"] = _run_original(
                                client, model, documents, instructions, max_tokens,
                                original_placeholder,
                            )
                            status.update(label="Original draft complete", state="complete")
                        if results.get("original_text"):
                            st.download_button(
                                "Download original (.md)",
                                results["original_text"].encode("utf-8"),
                                "claimant_rejoinder.md", "text/markdown",
                                use_container_width=True, key="dl_o_live",
                            )
                    except Exception as exc:
                        results["original_error"] = str(exc)
                        st.error(f"Original draft failed: {exc}")

                # --- Grounded draft (live stream) --------------------------
                with col_grounded:
                    try:
                        with st.status("Extracting graph & drafting (grounded)…", expanded=False) as status:
                            grounded_text, graph, packs, posture = _run_grounded(
                                client, model, documents, instructions, max_tokens,
                                use_example, grounded_placeholder,
                            )
                            status.update(label="Grounded draft complete", state="complete")
                        results.update({
                            "grounded_text": grounded_text, "graph": graph,
                            "packs": packs, "posture": posture,
                            "coverage": coverage_report(graph, packs),
                        })
                        if grounded_text:
                            st.download_button(
                                "Download grounded (.md)",
                                grounded_text.encode("utf-8"),
                                "claimant_rejoinder_grounded.md", "text/markdown",
                                use_container_width=True, key="dl_g_live",
                            )
                        _render_artifacts(results)
                    except Exception as exc:
                        results["grounded_error"] = str(exc)
                        st.error(f"Grounded draft failed: {exc}")

                # --- Auto-ingest into Neo4j (flush-all, then current case) --
                if results.get("graph") and ng.neo4j_configured():
                    try:
                        with st.spinner("Flushing Neo4j and loading the case graph…"):
                            cid = ng.case_id_of(results["graph"])
                            counts = ng.ingest_current(results["graph"], cid)
                        results["subgraph"] = {"case_id": cid, "counts": counts}
                    except Exception as exc:
                        results["neo4j_error"] = str(exc)

                st.session_state["results"] = results
                _render_graph_section(results)
                return

    # Non-submit reruns (e.g. interacting with the graph) re-render from state
    # so the drafted rejoinders stay on screen.
    results = st.session_state.get("results")
    if results:
        _render_columns_static(results)
        _render_graph_section(results)


if __name__ == "__main__":
    main()
