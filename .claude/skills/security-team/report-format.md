# Finding format

Return findings as a numbered list, most severe first. One finding per real
exploit path; merge duplicates. Every field is required.

```
### <n>. <one-line title>
- Severity: Critical | High | Medium | Low | Info   (rubric in SKILL.md)
- Where: path:line (and the function)
- Attacker: which adversary from threat-model.md
- Path: the concrete steps — what they send, what the code does, what they get.
  Quote the lines that make it possible.
- Evidence: a reproduction you ran locally (command + observed output), or
  "not reproduced" with why.
- Fix: the smallest change that removes the path, at the trust boundary.
- Invariants at risk: which invariants the fix could break, and how the fix
  avoids that. "None" only if you checked.
- Test: the test that fails before and passes after.
```

End with:

- **Checked and fine** — the areas you examined that hold up (one line each),
  so the lead knows what was covered.
- **Not reachable / accepted** — anything that looks bad but has no path, with
  the reason.

Do not pad. Three verified findings beat thirty speculative ones.
