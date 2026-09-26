<img src="assets/banner.jpg" alt="16mm film style banner of the shrug emoji guy shrugging against a dark neutral studio background" width="100%">

# ComfyUI-qwenimage21-explorations

Tinkering and research hub for the Qwen Image ecosystem

See [LLM wiki index](docs/wiki/index.md)

## Tests

- Python: `cd tests && pytest . --rootdir=.`, with ComfyUI's interpreter. Run
  from `tests/`: from the repo root pytest collects the root as a package and
  imports its `__init__.py`, which loads ComfyUI and takes a CUDA context for
  every test, so the whole suite errors while a job has the GPU full. The node
  tests still need ComfyUI importable beside this checkout and skip without it.
- The web extension: `bun test` (scoped to `tests/js` by `bunfig.toml`). The
  parity test holds `web/thinking_level.js` to the imagegen app's copy through
  a local `coderef/comfy-apps` link, and skips without one.
- Against a live heylook: `python scripts/check_presets_live.py --base-url URL`
  sends every stored preset's request as the expander plans it, one token
  each, and fails on any the server refuses.

## Licence

MIT, see LICENSE. The models, their system prompts and the upstream repo carry
their own licences.
