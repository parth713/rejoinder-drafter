"""Persist a grounded case graph into Neo4j and read it back for visualization.

Scope (per the agreed plan): a STANDALONE case subgraph on the user's own remote
Neo4j. Every node is namespaced with a `caseId` so re-loading a case is idempotent
and one case never touches another (or any other graph on the instance).

The graph dict shape is what grounding_pipeline produces:
    { "graph_metadata": {...},
      "nodes": [{ "id", "label", "props"{...,"provenance":[...]}, "flags":[...], "suppressed":bool }],
      "relationships": [{ "from", "type", "to" }] }
"""

from __future__ import annotations

import json
import os
import re

from neo4j import GraphDatabase

# Ontology label/relationship names from grounding/SKILL.md ("The ontology (base)")
# plus the documented extensions. Used as an allow-list before interpolating a
# label/type into Cypher (driver params can't parameterise labels or rel types).
ONTOLOGY_LABELS = {
    "Loan", "SecuredCreditor", "Borrower", "CoBorrower", "Guarantor",
    "LegalManager", "CollectionManager", "SecuredAsset", "Communication",
    "LegalProceeding", "Section138Proceeding", "ArbitrationProceeding",
    "SARFAESIProceeding",
    # extensions
    "InterimMeasure", "ProceduralOrder", "AttachedAsset", "ThirdParty",
}

_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Colour per ontology label for the node-link diagram.
LABEL_COLORS = {
    "Loan": "#f59e0b",
    "SecuredCreditor": "#2563eb",
    "Borrower": "#16a34a",
    "CoBorrower": "#22c55e",
    "Guarantor": "#0d9488",
    "SecuredAsset": "#a855f7",
    "Communication": "#64748b",
    "ArbitrationProceeding": "#dc2626",
    "Section138Proceeding": "#e11d48",
    "SARFAESIProceeding": "#9333ea",
    "LegalProceeding": "#ef4444",
    "InterimMeasure": "#0ea5e9",
    "ProceduralOrder": "#06b6d4",
    "AttachedAsset": "#c026d3",
    "ThirdParty": "#78716c",
}
_DEFAULT_COLOR = "#9ca3af"


# --------------------------------------------------------------------------- #
# Configuration / driver
# --------------------------------------------------------------------------- #
def _env():
    return (
        os.getenv("NEO4J_URI", "").strip(),
        os.getenv("NEO4J_USERNAME", "").strip(),
        os.getenv("NEO4J_PASSWORD", "").strip(),
        os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j",
    )


def neo4j_configured() -> bool:
    uri, user, password, _ = _env()
    return bool(uri and user and password)


def get_driver():
    uri, user, password, _ = _env()
    if not (uri and user and password):
        raise RuntimeError(
            "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME and "
            "NEO4J_PASSWORD in .env."
        )
    return GraphDatabase.driver(uri, auth=(user, password))


def _database() -> str:
    return _env()[3]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:80] or "case"


def case_id_of(graph: dict) -> str:
    """Stable per-case key: loan account if present, else a slug of the reference."""
    meta = graph.get("graph_metadata", {})
    acct = (meta.get("loan_account") or "").strip()
    if acct:
        return _slug(acct)
    return _slug(meta.get("case_reference") or meta.get("matter") or "case")


def _is_scalar(value) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _safe_props(props: dict) -> dict:
    """Neo4j stores only scalars / lists-of-scalars; JSON-encode anything nested."""
    safe = {}
    for key, value in (props or {}).items():
        if _is_scalar(value):
            safe[key] = value
        elif isinstance(value, list) and all(_is_scalar(v) for v in value):
            safe[key] = value
        else:
            safe[key] = json.dumps(value, ensure_ascii=False)
    return safe


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
def _build_params(graph: dict):
    """Validate + flatten a graph dict into (node_params, rel_params)."""
    node_params = []
    valid_ids = set()
    for node in graph.get("nodes", []):
        label = node.get("label", "")
        node_id = node.get("id", "")
        if not node_id or label not in ONTOLOGY_LABELS or not _LABEL_RE.match(label):
            continue  # skip unknown labels rather than risk a bad Cypher label
        valid_ids.add(node_id)
        node_params.append(
            {
                "label": label,
                "id": node_id,
                "props": _safe_props(node.get("props", {})),
                "flags": node.get("flags", []) or [],
                "suppressed": bool(
                    node.get("suppressed") or node.get("props", {}).get("present") is False
                ),
            }
        )

    rel_params = []
    for rel in graph.get("relationships", []):
        rtype = rel.get("type", "")
        src, dst = rel.get("from"), rel.get("to")
        if not _REL_TYPE_RE.match(rtype) or src not in valid_ids or dst not in valid_ids:
            continue
        rel_params.append({"type": rtype, "from": src, "to": dst})
    return node_params, rel_params


def _merge_nodes_rels(tx, case_id: str, node_params: list, rel_params: list):
    for np in node_params:
        tx.run(
            f"MERGE (n:`{np['label']}` {{id:$id, caseId:$cid}}) "
            "SET n += $props, n.caseId=$cid, n.nodeLabel=$label, "
            "n.flags=$flags, n.suppressed=$suppressed",
            id=np["id"], cid=case_id, props=np["props"], label=np["label"],
            flags=np["flags"], suppressed=np["suppressed"],
        )
    for rp in rel_params:
        tx.run(
            "MATCH (a {id:$from, caseId:$cid}), (b {id:$to, caseId:$cid}) "
            f"MERGE (a)-[:`{rp['type']}`]->(b)",
            **{"from": rp["from"], "to": rp["to"], "cid": case_id},
        )


def load_graph(graph: dict, case_id: str) -> dict:
    """Idempotently (re)load ONE case's subgraph (deletes only that caseId)."""
    node_params, rel_params = _build_params(graph)

    def _tx(tx):
        tx.run("MATCH (n {caseId:$cid}) DETACH DELETE n", cid=case_id)
        _merge_nodes_rels(tx, case_id, node_params, rel_params)

    driver = get_driver()
    try:
        with driver.session(database=_database()) as session:
            session.execute_write(_tx)
    finally:
        driver.close()
    return {"nodes": len(node_params), "relationships": len(rel_params)}


def flush_database() -> None:
    """Delete EVERY node (and relationship) in the target database.

    Destructive by design (the MVP keeps only the current case). NEO4J_* must
    point at the dedicated case-graph DB, never a shared graph.
    """
    driver = get_driver()
    try:
        with driver.session(database=_database()) as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
    finally:
        driver.close()


def ingest_current(graph: dict, case_id: str) -> dict:
    """Flush the whole DB, then write ONLY the current case. Returns counts."""
    node_params, rel_params = _build_params(graph)

    def _tx(tx):
        tx.run("MATCH (n) DETACH DELETE n")  # flush all previous graph info
        _merge_nodes_rels(tx, case_id, node_params, rel_params)

    driver = get_driver()
    try:
        with driver.session(database=_database()) as session:
            session.execute_write(_tx)
    finally:
        driver.close()
    return {"nodes": len(node_params), "relationships": len(rel_params)}


def clear_case(case_id: str) -> int:
    driver = get_driver()
    try:
        with driver.session(database=_database()) as session:
            result = session.run(
                "MATCH (n {caseId:$cid}) "
                "WITH collect(n) AS ns, count(n) AS deleted "
                "FOREACH (x IN ns | DETACH DELETE x) "
                "RETURN deleted",
                cid=case_id,
            )
            return result.single()["deleted"]
    finally:
        driver.close()


# --------------------------------------------------------------------------- #
# Read back (the viz renders from this, proving the round-trip)
# --------------------------------------------------------------------------- #
def read_case_subgraph(case_id: str):
    driver = get_driver()
    try:
        with driver.session(database=_database()) as session:
            node_rows = session.run(
                "MATCH (n {caseId:$cid}) "
                "RETURN n.id AS id, n.nodeLabel AS label, n.flags AS flags, "
                "n.suppressed AS suppressed, properties(n) AS props",
                cid=case_id,
            ).data()
            rel_rows = session.run(
                "MATCH (a {caseId:$cid})-[r]->(b {caseId:$cid}) "
                "RETURN a.id AS from, type(r) AS type, b.id AS to",
                cid=case_id,
            ).data()
    finally:
        driver.close()
    return node_rows, rel_rows


# --------------------------------------------------------------------------- #
# Visualization (neo4j-viz — official Neo4j visualization library)
# --------------------------------------------------------------------------- #
# Caption is chosen from the first matching property so nodes read meaningfully;
# falls back to the ontology label.
_CAPTION_KEYS = (
    "creditorName", "borrowerName", "coBorrowerName", "name", "type",
    "caseNo", "measureType", "loanAccountNo",
)


def render_case_html(case_id: str, show_suppressed: bool = False, height: str = "640px") -> str:
    """Render the case subgraph as interactive HTML via neo4j-viz `from_neo4j`.

    Reads the case straight out of Neo4j and lets neo4j-viz map labels, colours
    and relationships using Neo4j's own styling. Returns an HTML string for
    `st.components.v1.html`.
    """
    from neo4j_viz.neo4j import from_neo4j

    suppressed_filter = "" if show_suppressed else "WHERE coalesce(n.suppressed, false) = false "
    suppressed_filter_m = "" if show_suppressed else "WHERE coalesce(m.suppressed, false) = false "
    cypher = (
        f"MATCH (n {{caseId:$cid}}) {suppressed_filter}"
        f"OPTIONAL MATCH (n)-[r]->(m {{caseId:$cid}}) {suppressed_filter_m}"
        "RETURN n, r, m"
    )

    driver = get_driver()
    try:
        with driver.session(database=_database()) as session:
            graph = session.run(cypher, cid=case_id).graph()
    finally:
        driver.close()

    vg = from_neo4j(graph)
    # Prefer a human-readable caption per node (label + a key property).
    for node in vg.nodes:
        props = node.properties or {}
        label = props.get("nodeLabel") or (node.caption or "")
        detail = next((str(props[k]) for k in _CAPTION_KEYS if props.get(k)), "")
        node.caption = f"{label}: {detail[:24]}" if detail else label
    for rel in vg.relationships:
        if not rel.caption:
            rel.caption = rel.properties.get("type", "") if rel.properties else ""

    return vg.render(height=height).data
