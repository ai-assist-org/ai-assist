"""LLM-judge rubric for scoring RCA reports.

Ported verbatim from dci-mcp-server's tests/test_rca_eval.py (JUDGE_PROMPT),
which already calibrates this rubric against real DCI RCA reports.
"""

JUDGE_PROMPT = """\
You are an expert judge evaluating the quality of a Root Cause Analysis (RCA) \
report for a CI/CD job failure. Score the report on each criterion below \
using 0, 1, or 2.

## Scoring rubric

1. **causal_depth**: How deep is the causal chain?
   - 0: Stops at the first error (symptom only)
   - 1: 2-3 levels of "why"
   - 2: 4+ levels with evidence at each level

2. **evidence_quality**: Are claims backed by specific log evidence?
   - 0: No log citations or file references
   - 1: Some file references but vague
   - 2: Specific log lines or file content cited at each causal level

3. **adversarial_challenge**: Was an alternative hypothesis explored?
   - 0: No alternative hypothesis mentioned
   - 1: Alternative mentioned but not investigated with evidence
   - 2: Alternative from a different failure category, investigated with evidence for/against

4. **confidence_calibration**: Is confidence level stated with justification?
   - 0: No confidence level stated
   - 1: Confidence stated but criteria not explained
   - 2: Confidence stated with evidence-based justification (what evidence supports it, what's missing)

5. **report_structure**: Does the report have the required sections?
   Required: Job Information, Failure Symptom, Causal Chain, Root Cause, \
Contributing Factors, Confidence Level, Recommendations
   - 0: Fewer than 4 of the 7 sections
   - 1: 4-5 sections present
   - 2: 6-7 sections present

6. **must_gather_usage**: Was must_gather / cluster state data used?
   - 0: Not referenced at all
   - 1: Referenced but not integrated as evidence in the causal chain
   - 2: must_gather findings used as supporting or refuting evidence

7. **actionable_recommendations**: Are the recommendations useful?
   - 0: Missing or completely generic ("fix the bug")
   - 1: Some specific recommendations
   - 2: Concrete, actionable recommendations linked to the identified root cause

## Output format

Return ONLY a JSON object with this exact structure, no other text:
```json
{
  "scores": {
    "causal_depth": <0-2>,
    "evidence_quality": <0-2>,
    "adversarial_challenge": <0-2>,
    "confidence_calibration": <0-2>,
    "report_structure": <0-2>,
    "must_gather_usage": <0-2>,
    "actionable_recommendations": <0-2>
  },
  "explanations": {
    "causal_depth": "<one sentence>",
    "evidence_quality": "<one sentence>",
    "adversarial_challenge": "<one sentence>",
    "confidence_calibration": "<one sentence>",
    "report_structure": "<one sentence>",
    "must_gather_usage": "<one sentence>",
    "actionable_recommendations": "<one sentence>"
  },
  "overall_assessment": "<2-3 sentence summary of the report quality>"
}
```

## Report to evaluate

"""

CRITERIA = [
    "causal_depth",
    "evidence_quality",
    "adversarial_challenge",
    "confidence_calibration",
    "report_structure",
    "must_gather_usage",
    "actionable_recommendations",
]

CONSENSUS_PROMPT = """\
You are reviewing root cause analysis (RCA) reports written independently by \
several different AI models, all analyzing the SAME underlying CI/CD job \
failure with no visibility into each other's work.

Each report was already scored for internal quality in isolation (how well \
argued and well evidenced it is on its own). That scoring cannot catch a \
report that is confident, well-evidenced, and well-structured, but reaches \
the WRONG conclusion because the other reports agree with each other instead. \
Your job is to compare their conclusions, not re-judge their quality.

For each report, identify the root cause it ultimately settled on (the \
deepest actionable cause it names, not just the first proximate symptom).

Then:
1. Group reports that agree on the same underlying root cause (the same \
   actual mechanism/component as the ultimate cause — minor wording \
   differences don't matter).
2. If a report's explanation is a strict subset of another's (it stops at an \
   earlier point on the SAME causal chain without contradicting it), count \
   it as agreeing, but note it stopped shallower.
3. If a report names a genuinely different root cause (a different \
   component/mechanism as the deepest cause, not just an earlier stop on the \
   same chain), flag it as an outlier/disagreement.
4. Note which explanation (if any) is backed by the most direct, hard-to-\
   misinterpret evidence (e.g. an explicit error message) versus one that is \
   more inferred/circumstantial.

## Output format

Return ONLY a JSON object with this exact structure, no other text:
```json
{
  "explanations": {
    "<model_name>": "<one sentence: the deepest root cause this report identifies>"
  },
  "agreement_groups": [
    {
      "models": ["<model_name>", "..."],
      "shared_root_cause": "<one sentence>",
      "depth_note": "<e.g. 'all reach the same depth' or '<model_name> stops one level shallower'>"
    }
  ],
  "outliers": [
    {
      "model": "<model_name>",
      "divergent_root_cause": "<one sentence>",
      "reason": "<why this contradicts rather than just stops short of the majority>"
    }
  ],
  "strongest_evidence": "<which group/explanation has the most direct evidence, and why>",
  "summary": "<2-3 sentence overview of the consensus situation>"
}
```

## Reports to compare

"""
