<img src="assets/banner.jpg" alt="16mm film style banner of the shrug emoji guy shrugging against a dark neutral studio background" width="100%">

# ComfyUI-qwenimage21-explorations

Tinkering and research hub for the Qwen Image ecosystem

See [LLM wiki index](docs/wiki/index.md)

## Example workflows

`example_workflows/` has a text-to-image and an edit graph that expand the
prompt through a heylook-served expander before encoding. Regenerate or verify
them with:

```
python scripts/build_example_workflows.py [--check] [--server http://127.0.0.1:8188]
```

`--check` compares the files on disk against the generator. `--server` also
validates node types, input names and widget counts against a running ComfyUI,
which is the only way to catch a schema change upstream.

## Tests

```
uv run --with pytest python -m pytest tests/ -q
```

## Licence

MIT, see LICENSE. The models, their system prompts and the upstream repo carry
their own licences.
