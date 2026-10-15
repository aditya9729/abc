// Synthetic CPU DOM/media fixtures. No browser or native performance evidence.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const template = fs.readFileSync(process.env.R1LITE_VIEWER_TEMPLATE || path.join(__dirname, '../../abc_bench/static/original-methods/index.html'), 'utf8');
const script = template.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/refresh\(\);\s*setInterval\(refresh, 30000\);\s*$/, '');
const keys = ['rtc', 'initialized_full_expo', 'matched_initialized_full_expo', 'trained_full_expo', 'continued_full_expo'];
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

const accepted = JSON.parse(fs.readFileSync(path.join(__dirname, '../fixtures/accepted_four_cohort_results.json'), 'utf8'));
const historicalFields = ['r1lite', 'r1lite_initialized_full_expo', 'r1lite_matched_initialized_full_expo', 'r1lite_trained_full_expo'];
function history() {return historicalFields.map(key => structuredClone(accepted[key]));}
function continuation() {
  const value = structuredClone(accepted.r1lite_trained_full_expo);
  Object.assign(value, {schema: 'sadhana.r1lite_continued_full_expo_viewer/1',
    run_id: 'explicit-synthetic-future-evaluation', evaluation_checkpoint_step: 10509,
    learning_updates: 50, successes: 2, comparison_binding_sha256: 'b'.repeat(64),
    producer_history_scope: {resumed_segment_complete: true, parent_history_complete: false,
      metrics_scope: 'final_logged_update_only', approximate_restart: true, new_learning_updates: 20},
    producer_update_counters: {actor_train_state: 50, edit_actor: 50, temp: 50,
      critic: 1000, filter_critic: 1000, batch_encoder: 1000},
    scope: 'Explicit synthetic CPU fixture; not an actual score', supervision: 'Synthetic fixture'});
  value.episodes = value.episodes.map((row, index) => ({...row, success: index < 2,
    outcome: index < 2 ? 'native_success' : 'native_timeout', video: (1000 + index).toString(16).padStart(16, '0') + '.mp4'}));
  return value;
}
test('four actual historical cohorts keep scores and original comparison with continuation absent', () => {
  const p = page(), data = history(), before = JSON.stringify(data);
  p.render(...data);
  assert.match(p.elements.get('r1lite-table-score').textContent, /8\/20/);
  assert.match(p.elements.get('r1lite-trained-score').textContent, /0\/20/);
  assert.match(p.elements.get('r1lite-comparison').textContent, /0\/20 \(0 percentage points\)/);
  assert.match(p.elements.get('r1lite-continued-score').textContent, /pending/);
  assert.equal(p.elements.get('r1lite-cohort').options[4].disabled, true);
  assert.equal(JSON.stringify(data), before);
});
test('new slot preserves old comparison and distinguishes optimizer counts from checkpoint label', () => {
  const p = page(), data = [...history(), continuation()], before = JSON.stringify(data);
  p.render(...data);
  const picker = p.elements.get('r1lite-cohort'); picker.value = 'continued_full_expo'; picker.onchange();
  assert.match(p.elements.get('r1lite-score').textContent, /50 policy \/ 1000 critic updates/);
  assert.match(p.elements.get('r1lite-score').textContent, /checkpoint label 10509/);
  assert.match(p.elements.get('r1lite-score').textContent, /2\/20/);
  assert.match(p.elements.get('r1lite-method').textContent, /final logged-update metrics only/);
  assert.match(p.elements.get('r1lite-method').textContent, /parent history remains incomplete/);
  assert.match(p.elements.get('r1lite-method').textContent, /approximate restart/);
  assert.match(p.elements.get('r1lite-comparison').textContent, /0\/20 \(0 percentage points\)/);
  assert.equal(JSON.stringify(data), before);
  for (const [index, key] of keys.entries()) {
    picker.value = key; picker.onchange();
    assert.equal(p.elements.get('r1lite-episode').children.length, 20);
    assert.equal(p.elements.get('r1lite-video').src, data[index].episodes[0].video);
  }
});
test('refresh retains fifth-cohort episode and playback position', () => {
  const p = page(), data = [...history(), continuation()]; p.render(...data);
  const picker = p.elements.get('r1lite-cohort'); picker.value = 'continued_full_expo'; picker.onchange();
  const episodes = p.elements.get('r1lite-episode'); episodes.value = '9'; episodes.onchange();
  const video = p.elements.get('r1lite-video'); video.currentTime = 7; const loads = video.loads;
  p.render(...data);
  assert.equal(picker.value, 'continued_full_expo'); assert.equal(episodes.value, '9');
  assert.equal(video.currentTime, 7); assert.equal(video.loads, loads);
});
for (const defect of ['whole_history', 'exact_resume', 'all_losses', 'counter', 'seed']) test('continued scope refuses ' + defect, () => {
  const p = page(), value = continuation();
  if (defect === 'whole_history') value.producer_history_scope.parent_history_complete = true;
  if (defect === 'exact_resume') value.producer_history_scope.approximate_restart = false;
  if (defect === 'all_losses') value.producer_history_scope.metrics_scope = 'all_updates';
  if (defect === 'counter') value.producer_update_counters.critic = 999;
  if (defect === 'seed') value.episodes[0].seed = 30013;
  assert.throws(() => p.render(...history(), value));
});
