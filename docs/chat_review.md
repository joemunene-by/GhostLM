# Reviewing teacher-written chat data

Every conversation a teacher model writes for GhostLM is read in full before it can be used for training. `scripts/review_chat.py next` hands out a batch; the reviewer writes one verdict per conversation; `scripts/review_chat.py apply` files the result.

## What to check, in every conversation, every line

Read each user and assistant turn completely, character by character. Do not skim and do not sample.

1. **Facts.** Every claim must be true. Check numbers, dates, names, versions, ports, event IDs, CVE details, units and definitions. If you are not certain a claim is right, treat it as wrong.
2. **Maths.** Redo every calculation yourself. The stated answer and every intermediate step must be correct.
3. **Code, commands and configs.** Check that they would actually run and do what the reply says: syntax, flags, quoting and escaping, imports, API names, and edge cases the reply claims to handle. Check that inline comments showing output (`# 8.0467`) are right.
4. **Answers the question.** The first sentence answers what was actually asked. No drifting to a different question, no padding.
5. **Consistency.** The reply never contradicts itself or corrects itself mid-answer. Later turns stay consistent with earlier ones, and "it/that" refers to the right thing.
6. **Natural user turns.** The user's message reads like a real person. Drop or fix conversations where a job or persona has been forced into a question it doesn't fit (for example an algebra problem framed as a "pentest scoring formula").
7. **Tone.** Plain and direct. No filler openers or closers, no emoji, no moralising, and no needless authorisation caveats on harmless or public-information questions.
8. **Identity.** GhostLM never names another model or company as itself.
9. **Punctuation.** No em dashes or en dashes. Straight quotes.
10. **Safety.** No working malware, no operational help with clearly harmful requests. Legitimate security education and defence are fine and should not be refused.

## Verdicts

Write one JSON object per line to the verdicts file:

- `{"id": "...", "verdict": "keep"}`: the conversation is correct and good as written.
- `{"id": "...", "verdict": "fix", "turns": [...], "reason": "..."}`: a small, certain correction makes it right (a wrong number, a broken flag, a stray dash). `turns` is the complete corrected conversation. Only fix what you are sure of.
- `{"id": "...", "verdict": "drop", "reason": "..."}`: anything that needs more than a small fix, or that you cannot verify.

When in doubt, drop. A smaller clean dataset trains a better model than a bigger one with errors in it.
