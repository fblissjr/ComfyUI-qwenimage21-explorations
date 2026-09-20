<img src="assets/banner.jpg" alt="16mm film style banner of the shrug emoji guy shrugging against a dark neutral studio background" width="100%">

# ComfyUI-qwenimage21-explorations

Tinkering and research hub for the Qwen Image ecosystem

See [LLM wiki index](docs/wiki/index.md)

## Example workflows

`example_workflows/` has a text-to-image and an edit graph that expand the
prompt through a heylook-served expander before encoding. Regenerate or verify
them with:

```
python scripts/build_example_workflows.py [--check]
```

## Tests

```
uv run --with pytest python -m pytest tests/ -q
```

## Licence

MIT, see LICENSE. The models, their system prompts and the upstream repo carry
their own licences.
