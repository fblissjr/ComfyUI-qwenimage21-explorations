# prompt_bank/

Prompts in the shape the expanders actually emit, for driving renders without a
live expander and for holding text fixed across arms.

**Why this exists.** A brief ("change the hat to red") and a rewritten prompt
are different registers, and the conditioning encoder only ever sees the
second. Feeding it a brief is off-distribution, which quietly invalidates any
comparison made that way.

Frontmatter carries `task`, `n_images`, the contract fields the answer declared,
and `source`:

- **`source: generated`** — real expander output, captured from a run and
  edited in no way. These *are* the distribution, and they are the yardstick.
- **`source: hand-written`** — written here to the shape of a named generated
  entry, because no expander run existed for the reference images in question.
  Held to the generated ones, not to taste.

What the shapes look like, read off the generated entries:

| mode | register |
|---|---|
| t2i | third-person description of the finished image, opening on medium and style, then subject, then spatial anchoring, materials, and light |
| edit, one image | imperative. What changes, then an explicit list of what must not, then a sentence holding rendering and framing constant. **No `<imageN>` tags** — the contract forbids them at one image |
| edit, two or more | `<imageN>` tags throughout, each image's **role stated** (which is the canvas, which supplies material), then placement, then per-image preservation clauses |

The contract those obey is [`../docs/wiki/contract.md`](../docs/wiki/contract.md),
and the authority behind it ships inside the checkpoints rather than here.

**The hand-written entries name reference images this repo does not ship.**
Their `note:` fields cite bare filenames that live in whatever directory the
running ComfyUI resolves `LoadImage` against, so a fresh checkout has the text
and not the pictures. The text is the point; substitute your own references and
the structure still holds. Nothing here stores a path.

**The generated edit entries name their upstream sample images** in a
`reference:` field, bare filenames in `<image1>` order, comma-separated. They
are the upstream repo's example images, under
`coderef/Qwen-Image-2.1/prompt_rewrite/data/images/`, and
`scripts/steps_sweep.py --task edit --ref-dir` reads them from there.
