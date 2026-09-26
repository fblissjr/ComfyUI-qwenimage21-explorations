// The expander's model and preset dropdowns: what to list and what each entry
// writes to the node's `model` and `preset` inputs. Pure, like
// thinking_level.js. The imagegen app lists them the same way (`modelsFor`).

// A model the node can be pointed at for a task. Edit sends the references, so
// it needs a model that serves vision (capabilities, not modalities); text to
// image is left open. Blank is the node's trained expander for the task.
export function modelOptions(served, task, current) {
    const fit = task === "edit" ? served.filter((m) => (m.capabilities ?? []).includes("vision")) : served;
    const ids = fit.map((m) => m.id).filter(Boolean).sort((a, b) => a.localeCompare(b));
    const options = [{ value: "", label: "(trained expander for the task)" }, ...ids.map((id) => ({ value: id, label: id }))];
    const cur = String(current ?? "").trim();
    if (cur && !ids.includes(cur)) {
        const why = served.some((m) => m.id === cur) ? "no vision, needed for edit" : "not served";
        options.push({ value: cur, label: `${cur} (${why})` });
    }
    return options;
}

// By id first, then by name case-insensitively, as the node finds a preset.
export function findPreset(presets, wanted) {
    const want = String(wanted ?? "").trim();
    if (!want) return undefined;
    const byName = presets.filter((p) => (p.name ?? "").toLowerCase() === want.toLowerCase());
    return presets.find((p) => p.id === want) ?? (byName.length === 1 ? byName[0] : undefined);
}

// Each preset by name, which is what a person reads in a saved graph. A name
// the server holds twice is written as the id, since the node refuses an
// ambiguous name.
export function presetOptions(presets, current) {
    const count = new Map();
    for (const p of presets) {
        const k = (p.name ?? "").toLowerCase();
        count.set(k, (count.get(k) ?? 0) + 1);
    }
    const listed = presets
        .filter((p) => p.id || p.name)
        .map((p) => (count.get((p.name ?? "").toLowerCase()) === 1 && p.name
            ? { value: p.name, label: p.name, id: p.id }
            : { value: p.id, label: `${p.name ?? "?"} (${p.id})`, id: p.id }))
        .sort((a, b) => a.label.localeCompare(b.label));
    const options = [{ value: "", label: "(none)" }, ...listed.map(({ value, label }) => ({ value, label }))];
    const cur = String(current ?? "").trim();
    if (!cur) return { options, selected: "" };
    const found = findPreset(presets, cur);
    const hit = found && listed.find((o) => o.id === found.id);
    if (hit) return { options, selected: hit.value };
    const many = presets.filter((p) => (p.name ?? "").toLowerCase() === cur.toLowerCase()).length > 1;
    options.push({ value: cur, label: `${cur} (${many ? "several presets have this name" : "not found"})` });
    return { options, selected: cur };
}
