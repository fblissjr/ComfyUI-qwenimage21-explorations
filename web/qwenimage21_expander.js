import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { modelOptions, presetOptions, findPreset } from "./expander_lists.js";
import { applyThinking, presetThinking, thinkingChoice, thinkingControls, thinkingOptions } from "./thinking_level.js";

// Dropdowns on the prompt expander for `model`, `preset` and thinking, filled
// from the heylook server at the node's `base_url` through this pack's
// `/qwenimage21/heylook/{models,presets}` routes. Thinking is one control, as
// heylook's chat panel has it, in place of the `thinking` toggle and the
// free-text `reasoning_effort`.
//
// The node's own inputs stay, hidden, and are what the graph sends: each
// dropdown is UI only (serialize: false) and writes its input, so saved
// workflows and API graphs are unchanged. Where a list cannot be had the text
// box comes back in place of its dropdown.
const NODE = "QwenImage21PEExpand";
// Lists are read again after this long, so a model loaded on the server shows up.
const FRESH_MS = 60_000;

const lists = new Map();
function lookup(kind, baseUrl) {
    const root = String(baseUrl ?? "").trim().replace(/\/+$/, "");
    if (!root) return Promise.resolve(null);
    const key = `${kind}|${root}`;
    const hit = lists.get(key);
    if (hit && Date.now() - hit.at < FRESH_MS) return hit.list;
    const list = api
        .fetchApi(`/qwenimage21/heylook/${kind}?${new URLSearchParams({ base_url: root })}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
        .then((j) => (kind === "models" ? j.data : j.presets) ?? [])
        .catch((err) => {
            // null is "could not ask", not "none": the text box comes back. Not cached.
            console.warn(`${NODE}: heylook ${kind} lookup failed`, err);
            lists.delete(key);
            return null;
        });
    lists.set(key, { at: Date.now(), list });
    return list;
}

// A UI-only dropdown standing in for `input`, at its place in the node. The
// frontend keys widgets by name, so it takes its own name and the input's label.
function standIn(node, input, name, label, onPick) {
    let byLabel = new Map();
    const w = node.addWidget("combo", name, "", (picked) => {
        const value = byLabel.get(picked);
        if (value !== undefined) onPick(value);
    }, { values: [], serialize: false });
    w.serialize = false;
    w.label = label;
    node.widgets.splice(node.widgets.indexOf(w), 1);
    node.widgets.splice(node.widgets.indexOf(input), 0, w);
    return {
        // Show `options` with `selected` chosen, or the text box when there is no list.
        show(options, selected) {
            input.hidden = !!options;
            w.hidden = !options;
            if (!options) return;
            byLabel = new Map(options.map((o) => [o.label, o.value]));
            w.options.values = options.map((o) => o.label);
            w.value = (options.find((o) => o.value === selected) ?? options[0]).label;
        },
    };
}

app.registerExtension({
    name: "qwenimage21_explorations.expander.dropdowns",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const spec = nodeData.input?.optional?.thinking ?? nodeData.input?.required?.thinking;
        const defaultThinks = spec?.[1]?.default !== false;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;
            const widget = (name) => node.widgets?.find((w) => w.name === name);
            const [model, preset, think, effort] = ["model", "preset", "thinking", "reasoning_effort"].map(widget);
            if (!model || !preset || !think || !effort) return r;
            const text = (w) => String(w.value ?? "").trim();

            let refresh = () => {};
            const set = (w, value) => {
                w.value = value;
                w.callback?.(value);
                refresh();
            };
            let lastServed = null;
            const modelPick = standIn(node, model, "model_pick", "model", (id) => {
                // Each model has its own words: a word the new one does not take
                // is dropped with the old choice, as the app does.
                const word = text(effort);
                const depth = lastServed?.find((m) => m.id === id)?.thinking?.depth;
                const { levels, aliases } = thinkingControls(id, lastServed ?? []);
                if (word && lastServed && (!depth || (depth.unknown !== "verbatim"
                    && !levels.includes(aliases[word] ?? word)))) effort.value = "";
                set(model, id);
            });
            const presetPick = standIn(node, preset, "preset_pick", "preset", (v) => set(preset, v));
            const thinkPick = standIn(node, think, "thinking_level", "thinking", (choice) => {
                const out = applyThinking(choice, think.value, text(effort), defaultThinks);
                think.value = out.thinking;
                effort.value = out.reasoning_effort;
                refresh();
            });
            effort.hidden = true;

            let asked = 0;
            refresh = async () => {
                const mine = ++asked;
                const base = widget("base_url")?.value;
                const [served, presets] = await Promise.all([lookup("models", base), lookup("presets", base)]);
                if (mine !== asked) return; // a later change is already on its way
                lastServed = served;

                modelPick.show(served && modelOptions(served, widget("task")?.value, text(model)), text(model));
                const p = presets && presetOptions(presets, text(preset));
                presetPick.show(p?.options, p?.selected);

                // The node takes a preset's thinking over its own input.
                const presetThinks = presetThinking(findPreset(presets ?? [], text(preset))?.params);
                const thinkingNow = presetThinks ?? think.value;
                const id = text(model);
                const word = text(effort);
                const known = served ?? [];
                let options = thinkingOptions(id, known, defaultThinks);
                let choice = thinkingChoice(id, thinkingNow, word, known, defaultThinks);
                // A typed word the model does not offer still goes, and heylook
                // refuses it: show it rather than a default that will not run.
                const c = thinkingControls(id, known);
                const depth = known.find((m) => m.id === id)?.thinking?.depth;
                if (word && depth && depth.unknown !== "verbatim" && (!c.sw || thinkingNow !== false)
                    && !c.levels.includes(c.aliases[word] ?? word)) {
                    choice = `level:${word}`;
                    options = [...options, { value: choice, label: `${word} (not offered by this model)` }];
                }
                if (presetThinks === false) {
                    const shown = options.find((o) => o.value === choice) ?? options[0];
                    options = [{ ...shown, label: `${shown.label} (preset: thinking off)` }];
                } else if (presetThinks === true) {
                    options = options.filter((o) => o.value !== "off");
                }
                thinkPick.show(options, choice);
                // Grow to fit a list that came back, never shrink what the user sized.
                const [w, h] = node.computeSize();
                node.setSize([Math.max(node.size[0], w), Math.max(node.size[1], h)]);
                node.setDirtyCanvas(true, true);
            };

            // Typed values (the text boxes, when a list is missing) and the
            // task, which decides which models can take an edit's references.
            for (const w of [widget("base_url"), model, preset, widget("task")]) {
                if (!w) continue;
                const cb = w.callback;
                w.callback = function () {
                    const out = cb?.apply(this, arguments);
                    refresh();
                    return out;
                };
            }
            // Saved values land after creation; read them once they have.
            const onConfigure = node.onConfigure;
            node.onConfigure = function () {
                const out = onConfigure?.apply(this, arguments);
                refresh();
                return out;
            };
            queueMicrotask(refresh);
            return r;
        };
    },
});
