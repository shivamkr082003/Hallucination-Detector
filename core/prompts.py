"""All prompts used by the detection pipeline."""

ANSWER_SYSTEM = (
    "You are a knowledgeable assistant. Answer the user's question directly and factually "
    "in 1-4 sentences of plain text. Do not use markdown, bullet points or headings."
)

# Step 3a: the generator plans its own verification (CoVe "plan" step) and
# reports a verbalized confidence. It sees its own answer here.
SELF_CHECK = """You answered a question. Now plan how to fact-check your own answer.

Question: {question}
Your answer: {answer}

1. Rate how confident you are that your answer is fully correct, from 0 to 100.
2. Write 2-3 short, specific fact-check questions that test the key factual claims in your answer
   (names, dates, numbers, places). Each question must make sense on its own, so do not refer to
   "the answer" or "the text".

Return JSON only: {{"confidence": <integer 0-100>, "verification_questions": ["...", "..."]}}"""

# Step 3b: the generator answers the fact-check questions WITHOUT seeing its
# original answer (factored CoVe), so errors are not simply copied over.
COVE_ANSWER = """Answer each question independently, accurately and briefly (one short sentence each).
If you are not sure, answer "unknown".

{questions}

Return JSON only: {{"answers": ["answer to question 1", "answer to question 2", ...]}}"""

# Step 3c: a neutral comparator checks whether the independent answers agree
# with the original answer. It must NOT use its own world knowledge.
COVE_JUDGE = """You are a strict consistency checker. Compare texts only; do NOT use your own knowledge
to decide what is true.

Original question: {question}
Original answer: {answer}

Fact-check questions and answers, produced separately by the same AI WITHOUT seeing the original answer:
{qa}

For each fact-check, label it:
- "consistent": its answer agrees with the original answer
- "inconsistent": its answer gives different facts (another name, number, date, place)
- "unclear": it is "unknown" or does not address the original answer's claims

Then give an overall verdict about the original answer:
- "supported": the fact-checks agree with it
- "partial": some agree, some disagree
- "contradicted": the fact-checks contradict its key claims
- "abstained": the original answer itself declines to answer, says it does not know, or says the
  question's premise is false

Return JSON only:
{{"checks": [{{"id": 1, "label": "consistent|inconsistent|unclear", "note": "max 12 words"}}],
  "verdict": "supported|partial|contradicted|abstained",
  "issues": ["short description of each problem found, empty list if none"]}}"""

# Step 2: SelfCheckGPT-style consistency check of each sentence against one
# stochastic sample.
CONSISTENCY_JUDGE = """You compare two answers that the same AI wrote for the same question.

Question: {question}

Reference passage (another independent answer):
\"\"\"{sample}\"\"\"

Sentences to check:
{sentences}

Label each sentence:
- "supported": the passage states or clearly implies the same facts
- "contradicted": the passage gives different facts (a different name, number, date, place or claim)
- "neutral": the passage does not address this sentence

Also give "overall": "agree" if the passage gives the same core answer to the question,
"partial" if it overlaps only in part, or "disagree" if its core answer is different.
Compare the texts only; do not use your own knowledge.

Return JSON only: {{"labels": [{{"id": 1, "label": "supported|contradicted|neutral"}}], "overall": "agree|partial|disagree"}}"""

# Evaluation only: grade an answer against a ground-truth reference.
CORRECTNESS_JUDGE = """Grade an AI answer against a reference answer.

Question: {question}
Reference answer: {reference}
AI answer: {answer}

Rules:
- If the reference says the question is UNANSWERABLE or has a false premise, the AI answer is
  correct ONLY if it clearly declines, says it does not know, or points out the false premise.
  Inventing an answer is incorrect.
- Otherwise, the AI answer is correct if its core answer matches the reference. Extra harmless
  detail is fine; any wrong key fact (name, date, number) makes it incorrect.

Return JSON only: {{"correct": true or false, "reason": "max 15 words"}}"""
