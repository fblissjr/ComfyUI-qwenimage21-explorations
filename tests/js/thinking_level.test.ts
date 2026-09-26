// The expander's one thinking dropdown (web/thinking_level.js), as heylook's
// chat panel has it: Default (named), Off where the model has a switch, then
// the model's words. It writes the node's `thinking` and `reasoning_effort`.
// thinking_parity.test.ts holds it to the imagegen app's implementation; these
// stand on their own where that app is not checked out.
import { test, expect, describe } from 'bun:test'
import { thinkingOptions, thinkingChoice, applyThinking, presetThinking } from '../../web/thinking_level.js'

const qwen = { values: ['xhigh', 'medium', 'low'], aliases: { high: 'xhigh' }, default: 'xhigh', unknown: 'raises', off: [] }
const withOff = { values: ['medium', 'none', 'low', 'xhigh'], aliases: {}, default: 'medium', unknown: 'ignored', off: ['none'] }
const deepseek = { values: ['high', 'max'], aliases: {}, default: null, unknown: 'ignored', off: [] }
const harmony = { values: ['medium'], aliases: {}, default: 'medium', unknown: 'verbatim', off: [] }
const served = [
  { id: 'qwen', capabilities: ['thinking', 'reasoning_effort'], thinking: { switch: 'enable_thinking', depth: qwen } },
  { id: 'with-off', capabilities: ['thinking', 'reasoning_effort'], thinking: { switch: 'enable_thinking', depth: withOff } },
  { id: 'deepseek', capabilities: ['thinking', 'reasoning_effort'], thinking: { switch: 'enable_thinking', depth: deepseek } },
  { id: 'harmony', capabilities: ['reasoning_effort'], thinking: { switch: null, depth: harmony } },
  { id: 'trained', capabilities: ['vision', 'thinking'], thinking: { switch: 'enable_thinking', depth: null } },
  { id: 'instruct', capabilities: ['vision'], thinking: { switch: null, depth: null } },
]
const labels = (model: string, dflt = true) => thinkingOptions(model, served, dflt).map((o) => o.label)

describe('the options', () => {
  test('the default word named, Off, then the words with the default marked', () => {
    expect(labels('qwen')).toEqual(['Default (xhigh)', 'Off', 'xhigh (default)', 'medium', 'low'])
  })
  test('a word that only turns thinking off is not offered twice', () => {
    expect(labels('with-off')).toEqual(['Default (medium)', 'Off', 'medium (default)', 'low', 'xhigh'])
  })
  test('on or off for a switch without depth, and with no model (the trained expander)', () => {
    expect(labels('trained')).toEqual(['Default (on)', 'Off'])
    expect(labels('')).toEqual(['Default (on)', 'Off'])
  })
  test('no Off without a switch, and more words where the template pastes any in', () => {
    expect(labels('harmony')).toEqual(['Default (medium)', 'medium (default)', 'low', 'high', 'xhigh'])
    expect(labels('instruct')).toEqual(['Default'])
  })
  test('On only where the default leaves thinking off and there is no default word', () => {
    expect(labels('trained', false)).toEqual(['Default (off)', 'Off', 'On'])
    expect(labels('deepseek')).toEqual(['Default (on)', 'Off', 'high', 'max'])
  })
  test('the union where nothing is known about a typed model', () => {
    expect(thinkingOptions('typed', [], true).map((o) => o.label)).toEqual(['Default (on)', 'Off', 'low', 'medium', 'high', 'xhigh'])
  })
})

describe('reading and writing the two inputs', () => {
  test('reads the choice back, an alias as its word and a stale word as the default', () => {
    expect(thinkingChoice('qwen', true, 'low', served, true)).toBe('level:low')
    expect(thinkingChoice('qwen', true, 'high', served, true)).toBe('level:xhigh')
    expect(thinkingChoice('qwen', false, 'low', served, true)).toBe('off')
    expect(thinkingChoice('qwen', true, 'max', served, true)).toBe('')
    expect(thinkingChoice('harmony', false, 'high', served, true)).toBe('level:high')
  })
  test('writes both inputs, keeping the word when thinking is turned off', () => {
    expect(applyThinking('off', true, 'low', true)).toEqual({ thinking: false, reasoning_effort: 'low' })
    expect(applyThinking('level:medium', false, 'low', true)).toEqual({ thinking: true, reasoning_effort: 'medium' })
    expect(applyThinking('', false, 'low', true)).toEqual({ thinking: true, reasoning_effort: '' })
  })
  test("a preset's say on thinking, under either spelling", () => {
    expect(presetThinking({ enable_thinking: false })).toBe(false)
    expect(presetThinking({ thinking: true })).toBe(true)
    expect(presetThinking({ temperature: 1 })).toBeUndefined()
  })
})
