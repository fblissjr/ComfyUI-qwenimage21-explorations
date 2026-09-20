# templates/

Local system-prompt variants, as markdown with YAML-ish frontmatter:

```
---
task: edit
note: why this variant exists
---

<system prompt body>
```

`task` must be `t2i` or `edit`.

**The two canonical prompts are not here.** They ship inside each checkpoint as
`system_prompt.txt` and are byte-identical to the upstream repo's
`prompt_rewrite/prompts/` copies (verified by sha256, 2026-09-20). Resolution
order in `src/qwenimage21_explorations/templates.py` mirrors upstream's:
explicit text, then a template file, then the checkpoint's own copy.

Keeping the canonical prompts out avoids a third copy that can drift, and the
drift failure is silent -- fluent output against the wrong answer contract.

Whether these models are prompt-agnostic at all is untested. A variant here is
a hypothesis, not a drop-in.
