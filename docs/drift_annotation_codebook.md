# Thematic drift annotation codebook

Annotate only the adaptive criterion content shown for each `blind_id`. Do not infer the generating
method or condition. A label describes the rubric text, not downstream judge performance.

Required fields:

- `blind_id`: copy unchanged.
- `domain_general`: `yes` only when the adaptive content applies across unrelated prompts and
  domains; otherwise `no`.
- `task_specificity`: `0` = fully domain-general; `1` = generally phrased but visibly influenced by
  example structure or vocabulary; `2` = specific to an observed task, domain, or content pattern.
- `copies_example_terms`: `yes` when distinctive example vocabulary or a close paraphrase appears;
  otherwise `no`.
- `duplicate_or_near_duplicate`: `yes` when adaptive criteria duplicate each other or the immutable
  core in practical meaning; otherwise `no`.
- `short_rationale`: one concise, evidence-based sentence, at most 300 characters.

Judge conditional language literally: a criterion that says "when relevant" need not apply to every
task. Mark structural influence at level 1 even if no exact term is copied. Mark level 2 when a
criterion assumes a particular workflow (for example, extracted fields or prior/next steps).

Annotators work independently. Do not consult condition metadata, downstream scores, judge output,
or another rater before initial submission. Disagreements may be resolved in a separate adjudicated
file after the independent files are hash-validated. Lexical flags can prioritize review but are
heuristics only and must never be copied into the human label as ground truth.
