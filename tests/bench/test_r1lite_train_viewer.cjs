// Explicit CPU DOM/media doubles; no browser, playback or native result evidence.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const template = fs.readFileSync(process.env.R1LITE_VIEWER_TEMPLATE || path.join(__dirname, '../../abc_bench/static/original-methods/index.html'), 'utf8');
const script = template.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/refresh\(\);\s*setInterval\(refresh, 30000\);\s*$/, '');
const history = JSON.parse(fs.readFileSync(path.join(__dirname, '../fixtures/accepted_five_cohort_results.json'), 'utf8'));
const historicalKeys = ['r1lite', 'r1lite_initialized_full_expo', 'r1lite_matched_initialized_full_expo', 'r1lite_trained_full_expo', 'r1lite_continued_full_expo'];
const modes = ['full', 'base', 'zero_edit'];

function page() {
  const elements = new Map();
  const element = tag => ({tag, children: [], dataset: {}, hidden: false, value: '', loads: 0,
    get options() {return this.children;}, append(...nodes) {this.children.push(...nodes);},
    replaceChildren(...nodes) {this.children = nodes;}, setAttribute(name, value) {this[name] = String(value);},
    pause() {this.paused = true;}, load() {this.loads += 1;}});
  const context = {document: {getElementById(id) {
    if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id);
  }, createElement: element}};
  for (const [id, values] of [['r1lite-cohort', ['rtc', 'initialized_full_expo', 'matched_initialized_full_expo', 'trained_full_expo', 'continued_full_expo']], ['r1lite-train-condition', modes]]) {
    const picker = element('select'); picker.value = values[0]; picker.children = values.map(value => ({value})); elements.set(id, picker);
  }
  vm.createContext(context); vm.runInContext(script, context);
  return {elements, old: () => context.renderR1Lite(...historicalKeys.map(key => history[key])), render: data => context.renderR1LiteTrainDiagnostics(data)};
}

function fixture() {
  return {schema: 'sadhana.expo_train_diagnostics_viewer/1', snapshot_utc: 'explicit CPU fixture',
    split: 'TRAIN', initialization_seed: 42, online_updates: 0, actor_weight_checkpoint_label: 2000,
    scope: 'Explicit synthetic fixture: three original modes, one initialization seed, differing random streams; no pooled benchmark or learning-gain claim.',
    conditions: Object.fromEntries(modes.map((condition, mode) => [condition, {
      condition, run_id: 'explicit-CPU-' + condition, successes: mode,
      episodes: Array.from({length: 4}, (_, i) => ({seed: 21000 + i, success: i < mode,
        outcome: i < mode ? 'native_success' : 'native_timeout', native_controls: i < mode ? 7 : 900,
        video: (10000 + mode * 10 + i).toString(16).padStart(16, '0') + '.mp4'})),
    }]))};
}

test('absent TRAIN field keeps its section hidden and all five historical cohorts intact', () => {
  const p = page(), before = JSON.stringify(history); p.old();
  const comparison = p.elements.get('r1lite-comparison').textContent;
  const oldSrc = p.elements.get('r1lite-video').src;
  p.render(undefined);
  assert.equal(p.elements.get('r1lite-train').hidden, true);
  assert.equal(p.elements.get('r1lite-cohort').options.length, 5);
  assert.equal(p.elements.get('r1lite-comparison').textContent, comparison);
  assert.equal(p.elements.get('r1lite-video').src, oldSrc);
  assert.equal(JSON.stringify(history), before);
});

test('three original modes show separate X/4 and exactly12 original-order scored selections', () => {
  const p = page(), data = fixture(), before = JSON.stringify(data); p.old(); p.render(data);
  const comparison = p.elements.get('r1lite-comparison').textContent;
  let visited = 0;
  for (const [mode, key] of modes.entries()) {
    const picker = p.elements.get('r1lite-train-condition'); picker.value = key; picker.onchange();
    assert.equal(p.elements.get('r1lite-train-' + key + '-score').textContent, mode + '/4 successes');
    assert.match(p.elements.get('r1lite-train-score').textContent, new RegExp(mode + '/4 TRAIN successes$'));
    const episodes = p.elements.get('r1lite-train-episode'); assert.equal(episodes.children.length, 4);
    for (let i = 0; i < 4; i++) {
      episodes.value = String(i); episodes.onchange();
      assert.match(episodes.children[i].textContent, new RegExp('TRAIN seed ' + (21000 + i)));
      assert.equal(p.elements.get('r1lite-train-video').src, data.conditions[key].episodes[i].video); visited++;
    }
  }
  assert.equal(visited, 12); assert.equal(JSON.stringify(data), before);
  assert.equal(p.elements.get('r1lite-comparison').textContent, comparison);
  assert.equal(p.elements.get('r1lite-cohort').options.length, 5);
  assert.match(template, /initialization seed42 · zero online updates/);
  assert.match(template, /Startup20999 is excluded/);
  assert.match(template, /Branch random streams can differ/);
  assert.match(template, /No pooled benchmark, DEV result or learning-gain claim/);
});

test('base bypass and zero edit retain distinct original meanings', () => {
  const p = page(); p.render(fixture());
  const picker = p.elements.get('r1lite-train-condition');
  picker.value = 'base'; picker.onchange();
  assert.match(p.elements.get('r1lite-train-method').textContent, /generates N32.*first base candidate/);
  picker.value = 'zero_edit'; picker.onchange();
  assert.match(p.elements.get('r1lite-train-method').textContent, /Q ranking with edit scale0/);
  assert.match(p.elements.get('r1lite-train-score').textContent, /ranking retained/);
});

test('unchanged refresh retains TRAIN mode, episode, playback and historical selection', () => {
  const p = page(), data = fixture(); p.old(); p.render(data);
  const oldPicker = p.elements.get('r1lite-cohort'); oldPicker.value = 'continued_full_expo'; oldPicker.onchange();
  const picker = p.elements.get('r1lite-train-condition'); picker.value = 'zero_edit'; picker.onchange();
  const episodes = p.elements.get('r1lite-train-episode'); episodes.value = '2'; episodes.onchange();
  const video = p.elements.get('r1lite-train-video'); video.currentTime = 12; const loads = video.loads;
  p.old(); p.render(data);
  assert.equal(oldPicker.value, 'continued_full_expo'); assert.equal(picker.value, 'zero_edit');
  assert.equal(episodes.value, '2'); assert.equal(video.currentTime, 12); assert.equal(video.loads, loads);
});

for (const defect of ['DEV', 'updates', 'seed', 'missing_mode', 'condition', 'run', 'startup', 'partial', 'score', 'filename', 'outcome', 'timeout']) test('refuses bad TRAIN scope: ' + defect, () => {
  const p = page(), data = fixture(), row = data.conditions.full.episodes[0];
  if (defect === 'DEV') data.split = 'DEV';
  if (defect === 'updates') data.online_updates = 1;
  if (defect === 'seed') data.initialization_seed = 43;
  if (defect === 'missing_mode') delete data.conditions.base;
  if (defect === 'condition') data.conditions.full.condition = 'base';
  if (defect === 'run') data.conditions.full.run_id = data.conditions.base.run_id;
  if (defect === 'startup') row.seed = 20999;
  if (defect === 'partial') data.conditions.full.episodes.pop();
  if (defect === 'score') data.conditions.full.successes = 1;
  if (defect === 'filename') row.video = '../unsafe.mp4';
  if (defect === 'outcome') row.outcome = 'native_success';
  if (defect === 'timeout') row.native_controls = 899;
  assert.throws(() => p.render(data));
});
