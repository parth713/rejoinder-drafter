---
name: Loan Arbitration Graph Grounding
description: >
  Turn an Indian loan-arbitration case record into a grounded knowledge graph and
  hand an LLM only the subgraph it needs to draft the Claimant's rejoinder. Use
  when extracting graph nodes from a loan-arbitration file, grounding a rejoinder
  on case data, or building the node relevance map. Trigger even if the user only
  says "extract graph nodes from this loan case", "ground the rejoinder on the
  case data", "build the node relevance map", or names a lender (Tata Capital,
  Piramal, etc.) plus a borrower dispute.
---

# Loan Arbitration Graph Grounding

Turn an Indian loan-arbitration case record into a grounded knowledge graph, then
hand an LLM **only** the subgraph it needs to draft the Claimant's rejoinder. The
goal is grounding: every averment traces to a node, every node traces to a source
PDF + page. This prevents the model from inventing collateral, guarantors, or
defences that do not exist in the file.

This skill was distilled from two live matters (Tata Capital v. Sinha; Piramal v.
Jain). They share a spine but diverge enough that **the pipeline is parameterised,
not fixed**. Read "Step 0 — Posture detection" first; it decides which branch runs.

---

## The ontology (base)

13 node types: `Loan`, `SecuredCreditor` (the lender, secured or not),
`Borrower`, `CoBorrower`, `Guarantor`, `LegalManager`, `CollectionManager`,
`SecuredAsset`, `Communication`, `LegalProceeding`, `Section138Proceeding`,
`ArbitrationProceeding`, `SARFAESIProceeding`.

18 relationships: `PROVIDED_BY`, `TAKEN_BY`, `HAS_CO_BORROWER`, `GUARANTEED_BY`,
`HAS_GUARANTOR`, `MANAGED_BY_LEGAL_MANAGER`, `MANAGED_BY_COLLECTION_MANAGER`,
`SECURED_BY`, `HAS_COMMUNICATION`, `HAS_LEGAL_PROCEEDING`,
`HAS_SECTION_138_PROCEEDING`, `HAS_ARBITRATION_PROCEEDING`,
`HAS_SARFAESI_PROCEEDING`, `INVOLVED_IN_LEGAL_PROCEEDING`,
`INITIATES_LEGAL_PROCEEDING`, `SUBJECT_OF_SARFAESI_PROCEEDING`,
`OVERSEES_LEGAL_PROCEEDING`, plus `BORROWER↔CO_BORROWER` / `BORROWER↔GUARANTOR`.

### Ontology extensions (add when the facts demand — they often do)

The base ontology has no home for interim relief, procedural orders, or attached
third-party assets. When the record contains them, add these and flag them as
extensions in the graph (`"_extension": true`):

| Extension node | When to add | Edge |
|---|---|---|
| `InterimMeasure` | A Section 17 (or Section 9) interim order exists | `(:ArbitrationProceeding)-[:HAS_INTERIM_MEASURE]->` |
| `ProceduralOrder` | Ex-parte order, cost order, adjournment with consequence | `(:ArbitrationProceeding)-[:HAS_PROCEDURAL_ORDER]->` |
| `AttachedAsset` | An asset (e.g. bank account) is frozen by an interim measure — **distinct from `SecuredAsset`/loan collateral** | `(:InterimMeasure)-[:ATTACHES_ASSET]->` |
| `ThirdParty` | A garnishee bank / non-party served with an order | `(:AttachedAsset)-[:HELD_BY_GARNISHEE]->` |

Do not force a frozen bank account into `SecuredAsset`: collateral and an
interim-attachment are different legal objects, and conflating them produces a
wrong pleading.

---

## Step 0 — Posture detection (run this first; it parameterises everything)

Read the respondent-side filing and the procedural orders, then classify on **two
independent axes**. Earlier versions collapsed these into a single "merits vs
gate" label, which mis-fires when a filing is informal yet the proceeding ran on
the merits (the Tata-Gupta held-out case). Filing-formality and proceeding-mode
are orthogonal — set both.

**Axis A — Filing formality** (what the respondent actually put in):
- `formal_SOD` — a Statement of Defence filed on the institution's record.
- `informal_representations` — an advocate's letter / email / postal reply, not a
  platform SOD.
- `no_filing` — silence.

**Axis B — Proceeding mode** (how the tribunal ran it) — *this axis selects the
branch*:
- `inter_partes` — both sides heard; the rejoinder rebuts on the merits.
- `ex_parte` — respondent defaulted and the tribunal proceeded ex-parte; the
  rejoinder leads with the maintainability/ex-parte gate and pleads merits only in
  the alternative ("without prejudice").

**Branch = f(Axis B), not f(Axis A).** Record both as
`graph_metadata.filing_formality` and `graph_metadata.proceeding_mode`, and the
resulting branch in `graph_metadata.procedural_posture`. The branch decides the
relevance map, the suppression rules, and the coverage checklist below.

Worked cells — the three live instances span the matrix and prove the axes are
independent:

| Instance | Axis A | Axis B | Branch |
|---|---|---|---|
| Tata-Sinha | `formal_SOD` | `inter_partes` | Merits |
| Piramal-Jain | `informal_representations` | `ex_parte` | Procedural-gate |
| Tata-Gupta | `informal_representations` | `inter_partes` | **Merits** |

When Axis A is `informal_representations` but the branch is Merits, check **which
lis the reply actually answers** and raise the matching flag — an informal letter
frequently is not joined to the SOC at all:
- `informal_reply_partial_traverse` — the letter answers the recall / commencement
  notice rather than the SOC, leaving some SOC averments technically unrebutted
  (Gupta).
- `reply_addresses_different_lis` — the letter answers a *different cause of action
  entirely* (e.g. a Section 25 Payment & Settlement Systems Act demand, or a
  cheque-dishonour notice), so neither the reply nor the rejoinder is joined to the
  SOC's claim. The SOC quantum then stands **unrebutted**, and the rejoinder may be
  litigating a notice that is not the claim (Agarwal). Surface this so the drafter
  does not treat the merits as joined.

Also detect, in either branch:
- **Co-borrower / guarantor present?** If yes, activate those nodes (do not leave
  them as "NA" boilerplate). **Corroboration rule:** a second respondent in the
  cause-title activates `CoBorrower` only when the SOA or agreement actually names
  a co-applicant / co-borrower. A bare "& othr" / "& anr" in the caption that the
  SOA contradicts ("Co-Applicant: NA") is boilerplate — keep `CoBorrower`
  suppressed and raise `phantom_co_respondent_in_caption`. (Piramal's Amit Jain
  was a real second respondent and activates; Agarwal's "& othr" was phantom and
  does not — the rule must tell them apart, so never key activation on the caption
  alone.)
- **Interim measure / account freeze present?** If yes, add the extension nodes.
- **Statement of Account present?** If absent, quantum nodes are thin — mark
  default attributes `"NOT IN RECORD"` rather than inferring numbers. If a Statement
  of Account is present but as-on a date **earlier** than the rejoinder, treat a
  higher rejoinder figure as a **stage progression**, not an unsupported number:
  record each stage snapshot in `Loan.quantumProgression` (amount, as-on date, and
  its own provenance), provenance the operative figure to the pleading where it
  appears, and raise `quantum_stage_progression` so the updated SOA is placed on
  record. Also record `Loan.quantumComposition` — the make-up of the operative
  figure (current overdue, penal, interest-accrued-at-termination, and
  balance-principal-on-acceleration). On a **recalled / accelerated** loan the
  claim is the *whole* outstanding (overdue + unmatured balance principal), not
  just the overdue installments; without the composition the drafter cannot defend
  acceleration against a "but only X EMIs are overdue" objection (Agarwal: Rs.
  1,16,748 = 25,422 overdue + 249 penal + 578 interest-at-termination + 90,499
  balance principal — only 25,422 is overdue, the rest is the accelerated body).
- **Conditional / procedural demands by the respondent?** A cost-shifting
  precondition ("the claimant must bear the arbitrator's fee"), a seat/venue
  objection ("Delhi only"), or a language demand are **neither** merits defences
  **nor** an ex-parte gate. If present, activate the **cost-allocation / seat**
  issue row (Step 3) and populate `ArbitrationProceeding.seat` and
  `ArbitrationProceeding.costAllocation`, answered from the arbitration clause of
  the agreement.

---

## Step 1 — Extract nodes with provenance

For each ontology node present in the record, populate only the attributes the
documents actually support, and attach a `provenance` array of `"<DOC> p.<n>"`.
Never invent an attribution; if a value is absent, write `null` or `"NOT IN
RECORD"`. Output one `loan_case_graph.json` with `nodes[]` and `relationships[]`.
(See the two worked instances shipped with this skill for the exact shape.)

Privacy: bank account numbers, IFSC, card numbers, and government IDs are
sensitive. Store them only where legally needed, mark them `[SENSITIVE]`, and
**withhold full values from the generative context** (Step 5).

---

## Step 2 — Issue extraction from the respondent's filing

Parse the SOD / reply / representations into discrete contentions. For each,
capture the plea verbatim-in-brief and its paragraph number. This list is the
spine of the relevance map.

---

## Step 3 — Issue → Node relevance map

Map each contention to the **minimum** nodes + attributes that answer it. The map
is not fixed across cases — regenerate it per matter. In the procedural-gate
branch, the ex-parte/maintainability row is the **first and dominant** issue;
merits rows are retrieved but tagged `"alternative_without_prejudice"` so the
draft does not concede the gate.

**Conditional row — cost-allocation / seat** (activate only when Step 0 detects a
respondent cost-shifting precondition or seat/venue objection): map it to
`ArbitrationProceeding.seat` + `ArbitrationProceeding.costAllocation`, answered
from the agreement's arbitration clause (which typically fixes cost on the
borrower and leaves seat/venue to the institution). Tag it `merits` — it is a
substantive rebuttal, not a gate — but keep it off the lead unless the respondent
made it a condition precedent, in which case it must be met head-on so the draft
does not appear to accept the precondition.

---

## Step 4 — Scoped retrieval (Neo4j Cypher)

Load the JSON (each node → `(:Label {id, ...props, provenance})`; each
relationship → typed edge). Retrieve **per issue, never whole-graph**. The
foundation, default-quantum, service-timeline, and appointment queries are
reusable across matters; two queries are **conditional**:

```cypher
// Foundation (always)
MATCH (l:Loan)-[:PROVIDED_BY]->(cr:SecuredCreditor)
MATCH (l)-[:TAKEN_BY]->(b:Borrower)
OPTIONAL MATCH (l)-[:HAS_CO_BORROWER]->(co:CoBorrower)
RETURN l, cr.creditorName, cr.nameChange, b.borrowerName, co.coBorrowerName, l.provenance;

// Appointment challenge (always — but read the actual mechanism, it varies)
MATCH (l:Loan)-[:HAS_ARBITRATION_PROCEEDING]->(a:ArbitrationProceeding)
RETURN a.appointmentMechanism, a.arbitrationInstitution, a.arbitratorName, a.provenance;

// CONDITIONAL — only in the procedural-gate branch
MATCH (a:ArbitrationProceeding)-[:HAS_PROCEDURAL_ORDER]->(p:ProceduralOrder)
RETURN p.orderType, p.orderDate, p.basis, p.claimantUse, p.provenance;

// CONDITIONAL — only when an interim measure exists
MATCH (a:ArbitrationProceeding)-[:HAS_INTERIM_MEASURE]->(m:InterimMeasure)
OPTIONAL MATCH (m)-[:ATTACHES_ASSET]->(as:AttachedAsset)-[:HELD_BY_GARNISHEE]->(g:ThirdParty)
RETURN m.measureType, m.orderDate, m.reliefGranted, as.assetType, g.name, m.provenance;

// Suppression guard — empties referenced only to plead "not applicable"
MATCH (n) WHERE n.id IN ['GUAR_NONE','ASSET_NONE','SARFAESI_NONE','S138_NONE'] RETURN n.id;
```

So: **same skeleton, but co-borrower must be OPTIONAL-matched and activated, and
the interim-measure / ex-parte queries are new and fire only when those facts
exist.** The queries are not identical across cases.

---

## Step 5 — Relevance / suppression rules

Carry these forward unchanged:
1. Drop empty nodes from generative context; pass them only as one-line "not
   applicable" facts.
2. Attribute-level scoping — hand the LLM only the attributes listed for the issue.
3. One issue, one pack — draft paragraph-by-paragraph.
4. Flags travel with facts.
5. No fact without provenance.

Add these (new since the Piramal matter):
6. **Co-borrower un-suppression** — when present, the co-borrower is relevant;
   plead the joint-and-several liability rather than dropping it.
7. **Gate-first suppression** — in the procedural-gate branch, suppress merits
   rebuttals from the *lead* paragraphs and elevate the ex-parte/maintainability
   objection; tag any merits content `without_prejudice` so the draft cannot
   waive the gate.
8. **Do-not-rebut-unserved** — if the Claimant disputes service of the
   respondent's representations, do not let the model engage their contents as if
   admitted; the pack must carry the "service disputed / strict proof" posture.
9. **Sensitive-identifier withholding** — never emit full bank-account numbers,
   IFSC, card, or government IDs into the draft; refer to them as "the account
   under freeze" with the order's provenance.

---

## Step 6 — Discrepancy flags (regenerate per case — do NOT reuse a fixed list)

Flags are case-specific. Always scan for, at minimum: arbitration-clause
reference mismatches; loan characterisation (business vs personal); outstanding
quantum vs as-on date; recall-notice boilerplate that doesn't fit the facility
(e.g. vehicle-surrender language on an unsecured loan); and **arbitrator-identity
consistency across the notice, SOC, interim order, and rejoinder** (Piramal named
Utkarsh Singh in the notice/SOC but Akshai Mani signed the interim order and
rejoinder — a chain-of-appointment gap the drafter must reconcile). Also flag any
respondent timeline that is internally impossible (Piramal: objections dated
before the invocation notice). Emit flags into each affected node and surface them
in the relevant pack.

---

## Step 7 — Output handed to the LLM (per-issue context pack)

Same schema across all cases:

```json
{
  "issue": "<plea + para no.>",
  "claimant_thrust": "<one line>",
  "rebuttal_nodes": [{ "id": "...", "props_used": { }, "provenance": [ ] }],
  "flags": ["..."],
  "posture_tag": "merits | gate_threshold | alternative_without_prejudice",
  "instruction": "Draft only from props_used. Assert nothing outside this pack. Cite provenance inline. Do not emit sensitive identifiers."
}
```

The template is identical; only two fields are new vs the first matter:
`posture_tag` (drives gate-first ordering) and the sensitive-identifier clause in
`instruction`. Everything else (issue, props_used, provenance, flags) is reused.

---

## Step 8 — Coverage check

Structurally identical principle every time: every respondent plea maps to ≥1
rebuttal node; every node resolves to a source PDF + page. But the **checklist
items are case-specific**, so regenerate them:
- Confirm the branch's dominant battlegrounds are each backed by a provenance-
  tagged subgraph. (Tata: quantum, service, institutional-appointment. Piramal:
  ex-parte gate, arbitrator independence, S.17 freeze justification — quantum is
  thin.)
- Confirm every **activated** node (co-borrower, interim measure, ex-parte order,
  garnishee) resolves to provenance.
- Confirm every extension node is flagged `_extension` so reviewers see the
  ontology was stretched.
- Confirm empties are walled off from generation.

---

## Answering "is it the same as the last case?" (quick reference)

| Question | Answer |
|---|---|
| Same scoped Cypher? | **Skeleton yes; not identical.** Co-borrower becomes OPTIONAL/active; new conditional queries for interim-measure & ex-parte. |
| Same suppression rules? | **Core yes; +4 new** (co-borrower un-suppress, gate-first, do-not-rebut-unserved, sensitive-ID withholding). |
| Same discrepancy flags? | **No — regenerate.** New flag class: arbitrator-identity chain; impossible respondent timeline. |
| Same LLM output schema? | **Yes; +2 fields** (`posture_tag`, sensitive-ID instruction). |
| Same coverage check? | **Same principle; different checklist** + must verify extension nodes. |

---

## Files shipped with this skill

- `references/tata_case_graph.json` — merits-SOD worked instance.
- `references/piramal_case_graph.json` — procedural-gate instance with the four
  ontology extensions populated.

Read whichever posture matches the matter at hand before extracting.
