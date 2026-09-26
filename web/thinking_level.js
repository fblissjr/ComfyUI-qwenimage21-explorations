// One control for the expander's thinking and its depth, as heylook's chat
// panel has it: Default (named for what it runs), Off where the model has a
// switch, On only where the default leaves thinking off and there is no default
// word to pick, then the model's own depth words with its default marked. It
// writes the node's two inputs, `thinking` and `reasoning_effort`.
//
// Pure, so it can be checked outside ComfyUI. The imagegen app carries the
// same rules (thinkingOptions / thinkingChoice / applyThinking); keep them in
// step.

// Offered only where the server gave no model's own words (a heylook before
// v2.0.95), and added where a template pastes any word in.
export const UNION = ["low", "medium", "high", "xhigh"];

// A models-route row: {id, capabilities, thinking: {switch, depth} | null}.
// No model is the node's trained expander: a switch and no depth.
export function thinkingControls(model, served) {
    const none = { sw: true, levels: [], defaultLevel: null, aliases: {} };
    if (!model) return none;
    if (!served.length) return { ...none, levels: UNION };
    const row = served.find((m) => m.id === model);
    if (!row) return none;
    const caps = row.capabilities || [];
    const sw = caps.includes("thinking");
    if (!caps.includes("reasoning_effort")) return { ...none, sw };
    if (!row.thinking) return { ...none, sw, levels: UNION };
    const depth = row.thinking.depth;
    if (!depth) return { ...none, sw };
    const off = depth.off || [];
    const own = (depth.values || []).filter((v) => !off.includes(v));
    const levels = depth.unknown === "verbatim" ? [...own, ...UNION.filter((v) => !own.includes(v))] : own;
    const defaultLevel = depth.default && levels.includes(depth.default) ? depth.default : null;
    return { sw, levels, defaultLevel, aliases: depth.aliases || {} };
}

// [{value, label}]; value is "" (Default), "off", "on" or "level:<word>".
export function thinkingOptions(model, served, defaultThinks) {
    const { sw, levels, defaultLevel } = thinkingControls(model, served);
    const named = sw ? (defaultThinks ? (defaultLevel ?? "on") : "off") : defaultLevel;
    return [
        { value: "", label: named ? `Default (${named})` : "Default" },
        ...(sw ? [{ value: "off", label: "Off" }] : []),
        ...(sw && !defaultThinks && !defaultLevel ? [{ value: "on", label: "On" }] : []),
        ...levels.map((v) => ({
            value: `level:${v}`,
            label: v === defaultLevel && (defaultThinks || !sw) ? `${v} (default)` : v,
        })),
    ];
}

// The choice the two inputs amount to. A word the model does not take reads as
// the default, which is what the node sends (it drops such a word).
export function thinkingChoice(model, thinking, effort, served, defaultThinks) {
    const { sw, levels, defaultLevel, aliases } = thinkingControls(model, served);
    const on = thinking !== false;
    if (sw && !on) return "off";
    const word = String(effort ?? "").trim();
    const canon = aliases[word] ?? word;
    if (word && levels.includes(canon)) return `level:${canon}`;
    if (!(sw && on && !defaultThinks)) return "";
    return defaultLevel ? `level:${defaultLevel}` : "on";
}

// {thinking, reasoning_effort} for a choice. Off keeps the word, so turning
// thinking back on returns to it.
export function applyThinking(choice, thinking, effort, defaultThinks) {
    if (choice === "off") return { thinking: false, reasoning_effort: effort };
    if (choice === "on") return { thinking: true, reasoning_effort: "" };
    if (choice.startsWith("level:")) return { thinking: true, reasoning_effort: choice.slice(6) };
    return { thinking: defaultThinks, reasoning_effort: "" };
}

// A preset's say on thinking, under either spelling stored presets use.
export function presetThinking(params) {
    const v = (params || {}).thinking ?? (params || {}).enable_thinking;
    return typeof v === "boolean" ? v : undefined;
}
