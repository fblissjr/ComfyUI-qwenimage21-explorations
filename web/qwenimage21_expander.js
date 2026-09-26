import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { applyThinking, presetThinking, thinkingChoice, thinkingControls, thinkingOptions } from "./thinking_level.js";

// One "thinking" dropdown on the prompt expander, as heylook's chat panel has
// it, in place of the `thinking` toggle and the free-text `reasoning_effort`.
// Both inputs stay on the node, hidden, and are what the graph sends: the
// dropdown is UI only (serialize: false), so saved workflows and API graphs
// are unchanged. Its words are the chosen model's own, read from heylook
// through this pack's `/qwenimage21/heylook/models` route.
const NODE = "QwenImage21PEExpand";
// Lists are read again after this long, so a model loaded on the server shows up.
const FRESH_MS = 60_000;

const lists = new Map();
function lookup(kind, baseUrl) {
    const root = String(baseUrl ?? "").trim().replace(/\/+$/, "");
    if (!root) return Promise.resolve([]);
    const key = `${kind}|${root}`;
    const hit = lists.get(key);
    if (hit && Date.now() - hit.at < FRESH_MS) return hit.list;
    const list = api
        .fetchApi(`/qwenimage21/heylook/${kind}?${new URLSearchParams({ base_url: root })}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
        .then((j) => (kind === "models" ? j.data : j.presets) ?? [])
        .catch((err) => {
            // Not cached: the next change asks again.
            console.warn(`${NODE}: heylook ${kind} lookup failed`, err);
            lists.delete(key);
            return [];
        });
    lists.set(key, { at: Date.now(), list });
    return list;
}

// By id first, then by name, as the node finds a preset.
function findPreset(presets, wanted) {
    const want = wanted.trim();
    if (!want) return undefined;
    return presets.find((p) => p.id === want)
        ?? presets.find((p) => (p.name ?? "").toLowerCase() === want.toLowerCase());
}

app.registerExtension({
    name: "qwenimage21_explorations.expander.thinking",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const spec = nodeData.input?.optional?.thinking ?? nodeData.input?.required?.thinking;
        const defaultThinks = spec?.[1]?.default !== false;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;
            const widget = (name) => node.widgets?.find((w) => w.name === name);
            const think = widget("thinking");
            const effort = widget("reasoning_effort");
            if (!think || !effort) return r;
            think.hidden = true;
            effort.hidden = true;

            let byLabel = new Map();
            // Its own name: the frontend keys widgets by name, and `thinking` is the toggle's.
            const level = node.addWidget("combo", "thinking_level", "", (label) => {
                const choice = byLabel.get(label);
                if (choice === undefined) return;
                const out = applyThinking(choice, think.value, String(effort.value ?? ""), defaultThinks);
                think.value = out.thinking;
                effort.value = out.reasoning_effort;
                node.setDirtyCanvas(true, true);
            }, { values: [], serialize: false });
            level.serialize = false;
            level.label = "thinking";

            let asked = 0;
            async function refresh() {
                const mine = ++asked;
                const base = widget("base_url")?.value;
                const model = String(widget("model")?.value ?? "").trim();
                const presetName = String(widget("preset")?.value ?? "");
                const [served, presets] = await Promise.all([
                    lookup("models", base),
                    presetName.trim() ? lookup("presets", base) : [],
                ]);
                if (mine !== asked) return; // a later change is already on its way
                // The node takes a preset's thinking over its own input.
                const presetThinks = presetThinking(findPreset(presets, presetName)?.params);
                const thinkingNow = presetThinks ?? think.value;
                const word = String(effort.value ?? "").trim();
                let options = thinkingOptions(model, served, defaultThinks);
                let choice = thinkingChoice(model, thinkingNow, word, served, defaultThinks);

                // A typed word the model does not offer still goes, and heylook
                // refuses it: show it rather than a default that will not run.
                const c = thinkingControls(model, served);
                const depth = served.find((m) => m.id === model)?.thinking?.depth;
                const reaches = !c.sw || thinkingNow !== false;
                if (word && depth && depth.unknown !== "verbatim" && reaches
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

                byLabel = new Map(options.map((o) => [o.label, o.value]));
                level.options.values = options.map((o) => o.label);
                level.value = (options.find((o) => o.value === choice) ?? options[0]).label;
                node.setDirtyCanvas(true, true);
            }

            for (const name of ["base_url", "model", "preset", "task"]) {
                const w = widget(name);
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
