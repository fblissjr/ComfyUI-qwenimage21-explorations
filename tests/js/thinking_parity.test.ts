// web/thinking_level.js is a port of the imagegen app's thinkingOptions /
// thinkingChoice / applyThinking (src/lib/heylook.ts). This holds the two to
// the same answers on every combination, so neither drifts alone. The app is
// found through the gitignored coderef/comfy-apps link; without it the test
// skips rather than failing a checkout that does not have the app.
import { test, expect } from 'bun:test'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import * as node from '../../web/thinking_level.js'

const APP = fileURLToPath(new URL('../../coderef/comfy-apps/src/lib/heylook.ts', import.meta.url))
const app = existsSync(APP) ? await import(APP) : null

test.skipIf(!app)('the node port matches the app on every case', () => {
  const served = app.parseModels({
    data: [
      { id: 'qwen', capabilities: ['thinking', 'reasoning_effort'], thinking: { switch: 'enable_thinking', depth: { values: ['xhigh', 'medium', 'low'], aliases: { high: 'xhigh' }, default: 'xhigh', unknown: 'raises', off: [] } } },
      { id: 'with-off', capabilities: ['thinking', 'reasoning_effort'], thinking: { switch: 'enable_thinking', depth: { values: ['medium', 'none', 'low', 'xhigh'], aliases: { off: 'none' }, default: 'medium', unknown: 'ignored', off: ['none'] } } },
      { id: 'deepseek', capabilities: ['thinking', 'reasoning_effort'], thinking: { switch: 'enable_thinking', depth: { values: ['high', 'max'], aliases: {}, default: null, unknown: 'ignored', off: [] } } },
      { id: 'harmony', capabilities: ['reasoning_effort'], thinking: { switch: null, depth: { values: ['medium'], aliases: {}, default: 'medium', unknown: 'verbatim', off: [] } } },
      { id: 'minimax', capabilities: ['vision', 'reasoning_effort'], thinking: { switch: null, depth: { values: ['enabled', 'disabled', 'adaptive'], aliases: {}, default: 'adaptive', unknown: 'fallback', off: [] } } },
      { id: 'trained', capabilities: ['vision', 'thinking'], thinking: { switch: 'enable_thinking', depth: null } },
      { id: 'instruct', capabilities: ['vision'], thinking: { switch: null, depth: null } },
      { id: 'old', capabilities: ['thinking', 'reasoning_effort'] },
    ],
  })
  const models = ['', 'qwen', 'with-off', 'deepseek', 'harmony', 'minimax', 'trained', 'instruct', 'old', 'typed']
  const efforts = ['', 'low', 'high', 'xhigh', 'max', 'none', 'off', 'medium', 'adaptive', 'bogus']
  for (const list of [served, []])
    for (const m of models)
      for (const dflt of [true, false]) {
        expect(node.thinkingOptions(m, list, dflt)).toEqual(app.thinkingOptions(m, list, dflt))
        for (const th of [true, false])
          for (const e of efforts) {
            const cfg = { model: m, thinking: th, reasoning_effort: e }
            expect(node.thinkingChoice(m, th, e, list, dflt)).toBe(app.thinkingChoice(cfg, list, dflt))
            for (const o of app.thinkingOptions(m, list, dflt)) {
              const a = app.applyThinking(cfg, o.value, dflt)
              expect(node.applyThinking(o.value, th, e, dflt)).toEqual({ thinking: a.thinking, reasoning_effort: a.reasoning_effort })
            }
          }
      }
})
