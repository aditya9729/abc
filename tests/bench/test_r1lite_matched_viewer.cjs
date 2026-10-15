// Synthetic CPU DOM/media fixtures. No browser or native performance evidence.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const template = fs.readFileSync(process.env.R1LITE_VIEWER_TEMPLATE || path.join(__dirname, '../../abc_bench/static/original-methods/index.html'), 'utf8');
const script = template.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/refresh\(\);\s*setInterval\(refresh, 30000\);\s*$/, '');
const keys = ['rtc', 'initialized_full_expo', 'matched_initialized_full_expo', 'trained_full_expo'];
function page() {
  const elements = new Map();
  const element = tag => ({tag, children: [], dataset: {}, hidden: false, value: '', loads: 0,
    get options() {return this.children;},
    append(...nodes) {this.children.push(...nodes);},
    replaceChildren(...nodes) {this.children = nodes;},
    setAttribute(name, value) {this[name] = String(value);},
    pause() {this.paused = true;}, load() {this.loads += 1;}});
  const context = {document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element(id));
      return elements.get(id);
    }, createElement: element,
  }};
  const picker = element('select'); picker.value = 'rtc';
  picker.children = keys.map(value => ({value})); elements.set('r1lite-cohort', picker);
  vm.createContext(context); vm.runInContext(script, context);
  return {elements, render: (...args) => context.renderR1Lite(...args)};
}
function cohort(key, successes) {
  const matched = key.startsWith('matched_') || key === 'trained_full_expo';
  const trained = key === 'trained_full_expo';
  return {schema: matched ? 'sadhana.r1lite_matched_full_expo_viewer/1' : key === 'rtc' ? 'sadhana.r1lite_viewer/1' : 'sadhana.r1lite_initialized_full_expo_viewer/1',
    run_id: 'synthetic-' + key, snapshot_utc: 'synthetic CPU fixture', successes,
    cohort: trained ? 'trained' : 'initialized', evaluation_checkpoint_step: trained ? 9909 : 0,
    learning_updates: trained ? 30 : 0, comparison_binding_sha256: 'a'.repeat(64),
    producer_history_scope: {acceptance_scope: 'frozen_evaluation_only', frozen_evaluation_only: true,
      raw_history_complete: false, training_metrics_available: false},
    scope: 'Synthetic fixture, no real result', supervision: key === 'rtc' ? 'Historical failed supervision' : 'Synthetic clean evaluation',
    episodes: Array.from({length: 20}, (_, i) => ({seed: 9100 + i, success: i < successes,
      outcome: i < successes ? 'native_success' : 'native_timeout', native_controls: 900,
      video: String(i + 1).padStart(16, '0') + '.mp4'}))};
}
function history() {return [cohort('rtc', 8), cohort('initialized_full_expo', 0)];}

test('missing cohorts stay hidden without changing another panel', () => {
  const p = page(); p.elements.set('catcher', {hidden: false}); p.render();
  assert.equal(p.elements.get('r1lite').hidden, true);
  assert.equal(p.elements.get('catcher').hidden, false);
});
test('historical40 remain available and matched table has no invented scores', () => {
  const p = page(); const data = history(), before = JSON.stringify(data); p.render(...data);
  assert.equal(p.elements.get('r1lite-table-score').textContent, '8/20 successes · 40%');
  assert.equal(p.elements.get('r1lite-initialized-score').textContent, '0/20 successes · 0%');
  assert.match(p.elements.get('r1lite-matched-initialized-score').textContent, /pending/);
  assert.match(p.elements.get('r1lite-trained-score').textContent, /pending/);
  assert.match(p.elements.get('r1lite-comparison').textContent, /pending/);
  assert.equal(JSON.stringify(data), before);
});
for (const available of ['before', 'after']) test('one new ' + available + ' cohort cannot claim improvement', () => {
  const p = page(); p.render(...history(), available === 'before' ? cohort('matched_initialized_full_expo', 2) : undefined,
    available === 'after' ? cohort('trained_full_expo', 4) : undefined);
  assert.match(p.elements.get('r1lite-comparison').textContent, /pending/);
  assert.doesNotMatch(p.elements.get('r1lite-comparison').textContent, /Observed success difference/);
});
test('both accepted matching fixtures show only the observed fixed-cohort difference', () => {
  const p = page(); p.render(...history(), cohort('matched_initialized_full_expo', 2), cohort('trained_full_expo', 4));
  assert.match(p.elements.get('r1lite-comparison').textContent, /\+2\/20 \(\+10 percentage points\)/);
  assert.match(p.elements.get('r1lite-comparison').textContent, /does not establish reliable improvement/);
});
test('all four selectors retain full original-order videos and truthful scopes', () => {
  const p = page(); const data = [...history(), cohort('matched_initialized_full_expo', 2), cohort('trained_full_expo', 4)];
  p.render(...data);
  const picker = p.elements.get('r1lite-cohort');
  for (const key of keys) {
    picker.value = key; picker.onchange();
    const episodes = p.elements.get('r1lite-episode');
    assert.equal(episodes.children.length, 20); assert.equal(episodes.value, '0');
    assert.match(episodes.children[0].textContent, /Seed 9100/);
    assert.equal(p.elements.get('r1lite-video').src, data[keys.indexOf(key)].episodes[0].video);
    if (key === 'trained_full_expo') assert.match(p.elements.get('r1lite-method').textContent, /Training loss metrics are unavailable/);
    if (key === 'initialized_full_expo') assert.match(p.elements.get('r1lite-method').textContent, /PRNG settings were not recorded/);
  }
});
test('refresh preserves selected new cohort, episode and media playback state', () => {
  const p = page(); const data = [...history(), cohort('matched_initialized_full_expo', 2), cohort('trained_full_expo', 4)];
  p.render(...data);
  const picker = p.elements.get('r1lite-cohort'); picker.value = 'trained_full_expo'; picker.onchange();
  const episodes = p.elements.get('r1lite-episode'); episodes.value = '17'; episodes.onchange();
  const video = p.elements.get('r1lite-video'); video.currentTime = 12; const loads = video.loads;
  p.render(...data);
  assert.equal(picker.value, 'trained_full_expo'); assert.equal(episodes.value, '17');
  assert.equal(video.currentTime, 12); assert.equal(video.loads, loads);
});
for (const damage of ['binding', 'scope', 'updates', 'seed']) test('invalid new cohort ' + damage + ' is refused', () => {
  const p = page(), before = cohort('matched_initialized_full_expo', 2), after = cohort('trained_full_expo', 4);
  if (damage === 'binding') after.comparison_binding_sha256 = 'b'.repeat(64);
  if (damage === 'scope') after.producer_history_scope.training_metrics_available = true;
  if (damage === 'updates') after.learning_updates = 0;
  if (damage === 'seed') after.episodes[19].seed = 9001;
  assert.throws(() => p.render(...history(), before, after));
});
