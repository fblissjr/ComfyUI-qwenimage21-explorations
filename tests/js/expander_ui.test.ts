// The expander's dropdowns (web/qwenimage21_expander.js) on a stand-in node:
// each sits in place of its input and writes it, is left out of what a
// workflow saves and a prompt sends (serialize: false), and gives way to the
// text box when the server cannot be asked. The extension imports ComfyUI's
// own `../../scripts/app.js`, so it runs from a copy under a temporary root
// with stubs there. The node's inputs are as the node declares them (0.5.0);
// the rows are shaped like the models route's.
import { test, expect, beforeAll } from 'bun:test'
import { cpSync, mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const depth = (values: string[], dflt: string | null, unknown = 'raises', aliases = {}) =>
  ({ values, aliases, default: dflt, unknown, off: [] })
const MODELS = [
  { id: 'Qwen-Image-2.1-PE-T21-mlx', capabilities: ['chat', 'vision', 'thinking'], thinking: { switch: 'enable_thinking', depth: null } },
  { id: 'qwen38', capabilities: ['chat', 'vision', 'thinking', 'reasoning_effort'], thinking: { switch: 'enable_thinking', depth: depth(['xhigh', 'medium', 'low'], 'xhigh') } },
  { id: 'qwen38-text', capabilities: ['chat', 'thinking', 'reasoning_effort'], thinking: { switch: 'enable_thinking', depth: depth(['xhigh', 'medium', 'low'], 'xhigh') } },
  { id: 'deepseek', capabilities: ['chat', 'thinking', 'reasoning_effort'], thinking: { switch: 'enable_thinking', depth: depth(['high', 'max'], null, 'ignored') } },
  { id: 'gpt-oss', capabilities: ['chat', 'reasoning_effort'], thinking: { switch: null, depth: depth(['medium'], 'medium', 'verbatim') } },
]
const PRESETS = [
  { id: 'a1', name: 'fun', params: { enable_thinking: true, reasoning_effort: 'medium' } },
  { id: 'b1', name: 'quiet', params: { enable_thinking: false } },
]
const INPUTS: [string, string, unknown][] = [
  ['task', 'COMBO', 't2i'], ['base_url', 'STRING', 'http://h'], ['model', 'STRING', ''],
  ['sampling', 'COMBO', 'reference'], ['brief', 'STRING', ''], ['preset', 'STRING', ''],
  ['checkpoint_dir', 'STRING', ''], ['local_template', 'COMBO', '(none)'], ['system_override', 'STRING', ''],
  ['thinking', 'BOOLEAN', true], ['timeout', 'INT', 900], ['max_pixels', 'INT', 1048576],
  ['reasoning_effort', 'STRING', ''],
]

let Node: any
beforeAll(async () => {
  const root = mkdtempSync(join(tmpdir(), 'qi21-ui-'))
  mkdirSync(join(root, 'scripts'))
  mkdirSync(join(root, 'ext', 'web'), { recursive: true })
  cpSync(fileURLToPath(new URL('../../web', import.meta.url)), join(root, 'ext', 'web'), { recursive: true })
  writeFileSync(join(root, 'scripts', 'app.js'), 'export const app = { exts: [], registerExtension(e) { this.exts.push(e) } }\n')
  writeFileSync(join(root, 'scripts', 'api.js'), `
    const MODELS = ${JSON.stringify(MODELS)}, PRESETS = ${JSON.stringify(PRESETS)}
    export const api = { async fetchApi(path) {
      const u = new URL(path, 'http://x')
      if (u.searchParams.get('base_url') !== 'http://h') return new Response('unreachable', { status: 502 })
      return Response.json(u.pathname.endsWith('/models') ? { data: MODELS } : { presets: PRESETS })
    } }\n`)
  const { app } = await import(join(root, 'scripts', 'app.js'))
  await import(join(root, 'ext', 'web', 'qwenimage21_expander.js'))
  Node = class {
    widgets: any[] = INPUTS.map(([name, type, value]) => ({ name, type, value, options: {} }))
    size = [400, 300]
    addWidget(type, name, value, callback, options) { const w = { type, name, value, callback, options }; this.widgets.push(w); return w }
    computeSize() { return [300, 26 * this.widgets.filter((w) => !w.hidden).length] }
    setSize(s) { this.size = s }
    setDirtyCanvas() {}
  }
  const input = { required: {}, optional: {} }
  for (const [name, type, value] of INPUTS) (input.optional as any)[name] = [type, { default: value }]
  await app.exts[0].beforeRegisterNodeDef(Node, { name: 'QwenImage21PEExpand', input })
})

const settle = () => new Promise((r) => setTimeout(r, 20))
const w = (n, name) => n.widgets.find((x) => x.name === name)
const pick = (n, name, label) => { const x = w(n, name); expect(x.options.values).toContain(label); x.callback(label) }
async function node() {
  const n = new Node()
  Node.prototype.onNodeCreated.call(n)
  await settle()
  return n
}

test('each dropdown stands in for its input, and none is saved or sent', async () => {
  const n = await node()
  for (const [ui, input] of [['model_pick', 'model'], ['preset_pick', 'preset'], ['thinking_level', 'thinking']]) {
    expect(n.widgets.indexOf(w(n, ui)) + 1).toBe(n.widgets.indexOf(w(n, input)))
    expect(w(n, input).hidden).toBe(true)
    expect(w(n, ui).serialize).toBe(false)
    expect(w(n, ui).options.serialize).toBe(false)
  }
  expect(w(n, 'reasoning_effort').hidden).toBe(true)
})

test("the thinking dropdown is the chosen model's, and writes both inputs", async () => {
  const n = await node()
  expect(w(n, 'thinking_level').options.values).toEqual(['Default (on)', 'Off'])
  pick(n, 'model_pick', 'qwen38'); await settle()
  expect(w(n, 'model').value).toBe('qwen38')
  expect(w(n, 'thinking_level').options.values).toEqual(['Default (xhigh)', 'Off', 'xhigh (default)', 'medium', 'low'])
  pick(n, 'thinking_level', 'low')
  expect([w(n, 'thinking').value, w(n, 'reasoning_effort').value]).toEqual([true, 'low'])
  pick(n, 'thinking_level', 'Off'); await settle()
  expect([w(n, 'thinking').value, w(n, 'reasoning_effort').value]).toEqual([false, 'low'])
  pick(n, 'model_pick', 'gpt-oss'); await settle()
  expect(w(n, 'thinking_level').options.values).toEqual(['Default (medium)', 'medium (default)', 'low', 'high', 'xhigh'])
})

test('a new model drops a word it does not take', async () => {
  const n = await node()
  pick(n, 'model_pick', 'qwen38'); await settle()
  pick(n, 'thinking_level', 'medium')
  pick(n, 'model_pick', 'deepseek'); await settle()
  expect(w(n, 'reasoning_effort').value).toBe('')
})

test('a saved word the model does not offer shows as such, not as a default', async () => {
  const n = await node()
  w(n, 'model').value = 'qwen38'
  w(n, 'reasoning_effort').value = 'max'
  n.onConfigure(); await settle()
  expect(w(n, 'thinking_level').value).toBe('max (not offered by this model)')
})

test('a preset that names thinking decides it', async () => {
  const n = await node()
  pick(n, 'preset_pick', 'quiet'); await settle()
  expect(w(n, 'preset').value).toBe('quiet')
  expect(w(n, 'thinking_level').options.values).toEqual(['Off (preset: thinking off)'])
  pick(n, 'preset_pick', 'fun'); await settle()
  expect(w(n, 'thinking_level').options.values).not.toContain('Off')
})

test('edit lists only vision models, and marks a text one already set', async () => {
  const n = await node()
  w(n, 'task').value = 'edit'
  w(n, 'model').value = 'qwen38-text'
  n.onConfigure(); await settle()
  expect(w(n, 'model_pick').options.values).not.toContain('deepseek')
  expect(w(n, 'model_pick').value).toBe('qwen38-text (no vision, needed for edit)')
})

test('without a server to ask, the text boxes come back', async () => {
  const n = await node()
  w(n, 'base_url').value = 'http://elsewhere'
  w(n, 'base_url').callback?.('http://elsewhere'); await settle()
  expect([w(n, 'model').hidden, w(n, 'model_pick').hidden]).toEqual([false, true])
  expect([w(n, 'preset').hidden, w(n, 'preset_pick').hidden]).toEqual([false, true])
})
