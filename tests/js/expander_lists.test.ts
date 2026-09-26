// The expander's model and preset dropdowns (web/expander_lists.js): what each
// lists and what a pick writes to the node's `model` and `preset` inputs.
import { test, expect } from 'bun:test'
import { modelOptions, presetOptions, findPreset } from '../../web/expander_lists.js'

const served = [
  { id: 'vl-b', capabilities: ['chat', 'vision'] },
  { id: 'text-a', capabilities: ['chat'] },
  { id: '', capabilities: [] },
]
const vals = (o: { value: string }[]) => o.map((x) => x.value)
const labels = (o: { label: string }[]) => o.map((x) => x.label)

test('t2i lists every served model, sorted, blank first', () => {
  expect(vals(modelOptions(served, 't2i', ''))).toEqual(['', 'text-a', 'vl-b'])
})
test('edit lists only models that serve vision', () => {
  expect(vals(modelOptions(served, 'edit', ''))).toEqual(['', 'vl-b'])
})
test('a current model not listed stays, saying why', () => {
  expect(labels(modelOptions(served, 'edit', 'text-a')).at(-1)).toBe('text-a (no vision, needed for edit)')
  expect(labels(modelOptions(served, 't2i', 'gone')).at(-1)).toBe('gone (not served)')
  expect(modelOptions(served, 't2i', 'vl-b').length).toBe(3)
})

const presets = [
  { id: 'a1', name: 'normal' },
  { id: 'c1', name: 'dupe' },
  { id: 'd1', name: 'DUPE' },
  { id: 'b1', name: 'Alpha' },
]
test('presets by name, a doubled name by id, sorted, none first', () => {
  const { options, selected } = presetOptions(presets, '')
  expect(options).toEqual([
    { value: '', label: '(none)' },
    { value: 'Alpha', label: 'Alpha' },
    { value: 'c1', label: 'dupe (c1)' },
    { value: 'd1', label: 'DUPE (d1)' },
    { value: 'normal', label: 'normal' },
  ])
  expect(selected).toBe('')
})
test('a current preset is found by id or by name, as the node finds it', () => {
  expect(presetOptions(presets, 'a1').selected).toBe('normal')
  expect(presetOptions(presets, 'NORMAL').selected).toBe('normal')
  expect(presetOptions(presets, 'd1').selected).toBe('d1')
})
test('an ambiguous or missing name stays, marked not found', () => {
  const r = presetOptions(presets, 'dupe')
  expect(r.selected).toBe('dupe')
  expect(r.options.at(-1)).toEqual({ value: 'dupe', label: 'dupe (several presets have this name)' })
  expect(findPreset(presets, 'dupe')).toBeUndefined()
})
