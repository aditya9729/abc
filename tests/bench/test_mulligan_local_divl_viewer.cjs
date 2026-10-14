// Explicit CPU DOM/media fixtures. No policies, native results, or browser proof.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const template = fs.readFileSync(path.join(__dirname, '../../abc_bench/static/original-methods/index.html'), 'utf8');
const script = template.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace(/refresh\(\);\s*setInterval\(refresh, 30000\);\s*$/, '');

function page() {
  const elements = new Map();
  const element = tag => ({tag, children: [], dataset: {}, hidden: false,
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; },
    setAttribute(name, value) { this[name] = value; },
    querySelectorAll(name) { return this.children.flatMap(node => [
      ...(node.tag === name ? [node] : []), ...(node.querySelectorAll ? node.querySelectorAll(name) : []),
    ]); }});
  const context = {document: {
    getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); },
    createElement: element, createTextNode: value => ({textContent: value}),
  }};
  vm.createContext(context);
  vm.runInContext(script, context);
  return {elements, render: value => context.renderMulligan(value)};
}

function clips() {
  return [0, 1, 2].map(point => {
    const sha256 = String(point + 1).padEnd(64, '0');
    return {point, complete: true, solved: point === 1, controls: 100, frames: 101,
      sha256, video: sha256.slice(0, 16) + '.mp4'};
  });
}

function fixture() {
  return {policies: [0, 1].map(n => ({label: 'Released CPU fixture ' + n, episodes: 50, successes: 25,
    clips: clips().map(c => ({...c, outcome: 'fixture', control_steps: c.controls}))})),
    pairing_text: 'CPU fixture', training_text: 'CPU fixture',
    local_training: {status: 'accepted_local_checkpoint', label: 'Existing IDQL fixture', summary: 'Preserved', clips: clips()},
    local_divl: {status: 'accepted_local_checkpoint', stage: 'divl_head', num_action_samples: 32,
      scores: [{n: 1, episodes: 50, successes: 25}, {n: 32, episodes: 50, successes: 20}], clips: clips()}};
}

test('absent optional DIVL keeps existing local IDQL and released panels', () => {
  const p = page(), value = fixture();
  delete value.local_divl;
  p.render(value);
  assert.equal(p.elements.get('mulligan-local-divl').hidden, true);
  assert.equal(p.elements.has('mulligan-local-divl-summary'), false);
  assert.equal(p.elements.get('mulligan-local-heading').textContent, 'Existing IDQL fixture');
  assert.equal(p.elements.get('mulligan-local-clips').children.length, 3);
  assert.equal(p.elements.get('mulligan-viewer').querySelectorAll('video').length, 2);
});

test('accepted full cohort renders separate stage scores and three fixed full videos', () => {
  const p = page();
  p.render(fixture());
  assert.equal(p.elements.get('mulligan-local-divl').hidden, false);
  assert.equal(p.elements.get('mulligan-local-divl-heading').textContent, 'Locally trained DIVL · N=32');
  assert.match(p.elements.get('mulligan-local-divl-summary').textContent, /N1 25\/50 · N32 20\/50/);
  assert.match(p.elements.get('mulligan-local-divl-summary').textContent, /does not isolate a critic effect/);
  assert.equal(p.elements.get('mulligan-local-divl-clips').children.length, 3);
  assert.equal(p.elements.get('mulligan-local-summary').textContent, 'Preserved');
  for (const [container, label] of [
    ['mulligan-local-clips', 'Existing IDQL fixture'],
    ['mulligan-local-divl-clips', 'Locally trained DIVL · N=32'],
  ]) {
    p.elements.get(container).querySelectorAll('video').forEach((video, point) => {
      assert.equal(video['aria-label'], label + ', start ' + point);
    });
  }
});

test('refresh preserves both stage video identities and their playback positions', () => {
  const p = page();
  p.render(fixture());
  const before = ['mulligan-local-clips', 'mulligan-local-divl-clips'].map(id => p.elements.get(id).querySelectorAll('video'));
  before.forEach(videos => { videos[0].currentTime = 1; });
  p.render(fixture());
  ['mulligan-local-clips', 'mulligan-local-divl-clips'].forEach((id, i) => {
    const after = p.elements.get(id).querySelectorAll('video');
    assert.equal(after[0], before[i][0]);
    assert.equal(after[0].currentTime, 1);
  });
});

test('pending/partial scores, wrong stage, and incomplete or foreign videos stay hidden', () => {
  const changes = [
    d => { d.status = 'running'; }, d => { d.stage = 'idql_agent'; },
    d => { d.num_action_samples = 1; }, d => { d.scores.pop(); },
    d => { d.scores[0].episodes = 49; }, d => { d.scores[0].n = 32; },
    d => { d.scores[0].successes = 51; }, d => { d.clips.pop(); },
    d => { d.clips[0].point = 3; }, d => { d.clips[0].complete = false; },
    d => { d.clips[0].solved = null; }, d => { d.clips[0].controls = 0; },
    d => { d.clips[0].frames--; }, d => { d.clips[0].sha256 = 'missing'; },
    d => { d.clips[0].video = '../foreign.mp4'; },
  ];
  for (const change of changes) {
    const p = page(), value = fixture();
    change(value.local_divl);
    assert.throws(() => p.render(value));
    assert.equal(p.elements.get('mulligan-local-divl').hidden, true);
    assert.equal(p.elements.has('mulligan-local-divl-summary'), false);
  }
});

function fullGrid(status = 'pending') {
  // Synthetic display contract only. These are not observed native outcomes.
  return {schema: 'mulligan.local-divl-full-grid/v1', status, task: 'square_narrow',
    run_id: 'author-subset-20261014T000000-000000', training_seed: 1,
    num_action_samples: 32, eval_seed: 42, env_seed: 42, episodes: 8000,
    completed_episodes: status === 'pending' ? 70 : 8000,
    successes: status === 'pending' ? null : 4000,
    ...(status === 'pending' ? {} : {receipt: {path: '/explicit/synthetic/receipt.json', bytes: 1, sha256: 'a'.repeat(64)}})};
}

test('absent full-grid record leaves the earlier fifty-start scores and videos', () => {
  const p = page();
  p.render(fixture());
  assert.equal(p.elements.get('mulligan-local-divl-full-grid').hidden, true);
  assert.equal(p.elements.get('mulligan-local-divl-full-grid-results').hidden, true);
  assert.match(p.elements.get('mulligan-local-divl-summary').textContent, /N1 25\/50 · N32 20\/50/);
});

test('pending full grid shows completed count without a success score or denominator', () => {
  const p = page(), value = fixture();
  value.local_divl.full_grid_evaluation = fullGrid();
  p.render(value);
  assert.equal(p.elements.get('mulligan-local-divl-full-grid').hidden, false);
  assert.equal(p.elements.get('mulligan-local-divl-full-grid-results').hidden, true);
  assert.match(p.elements.get('mulligan-local-divl-full-grid-status').textContent, /pending acceptance.*70\/8,000.*No full-cohort score/);
  assert.equal(p.elements.get('mulligan-local-divl-full-grid-rows').children.length, 0);
  value.local_divl.full_grid_evaluation.completed_episodes = 8000;
  p.render(value);
  assert.equal(p.elements.get('mulligan-local-divl-full-grid-results').hidden, true);
  assert.match(p.elements.get('mulligan-local-divl-full-grid-status').textContent, /pending acceptance.*8,000\/8,000/);
});

test('only a complete accepted grid displays synthetic successes and failures', () => {
  const p = page(), value = fixture();
  value.local_divl.full_grid_evaluation = fullGrid('accepted_complete');
  p.render(value);
  assert.equal(p.elements.get('mulligan-local-divl-full-grid-results').hidden, false);
  const row = p.elements.get('mulligan-local-divl-full-grid-rows').children[0];
  assert.deepEqual(row.children.map(cell => cell.textContent), ['8,000', '4,000', '4,000']);
  assert.equal(p.elements.get('mulligan-local-divl-full-grid').querySelectorAll('video').length, 0);
  for (const successes of [0, 8000]) {
    value.local_divl.full_grid_evaluation.successes = successes;
    p.render(value);
    assert.equal(p.elements.get('mulligan-local-divl-full-grid-results').hidden, false);
  }
});

test('full-grid refresh, completion and removal retain earlier players and positions', () => {
  const p = page(), value = fixture();
  p.render(value);
  const ids = ['mulligan-viewer', 'mulligan-local-clips', 'mulligan-local-divl-clips'];
  const originals = ids.map(id => p.elements.get(id).querySelectorAll('video'));
  originals.forEach(videos => { videos[0].currentTime = 2; });
  value.local_divl.full_grid_evaluation = fullGrid();
  p.render(value);
  value.local_divl.full_grid_evaluation = fullGrid('accepted_complete');
  p.render(value);
  p.render(structuredClone(value));
  value.local_divl.full_grid_evaluation = fullGrid();
  p.render(value);
  assert.equal(p.elements.get('mulligan-local-divl-full-grid-results').hidden, true);
  assert.equal(p.elements.get('mulligan-local-divl-full-grid-rows').children.length, 0);
  delete value.local_divl.full_grid_evaluation;
  p.render(value);
  ids.forEach((id, i) => {
    const videos = p.elements.get(id).querySelectorAll('video');
    assert.equal(videos[0], originals[i][0]);
    assert.equal(videos[0].currentTime, 2);
  });
  assert.equal(p.elements.get('mulligan-local-divl-full-grid').hidden, true);
});

test('wrong full-grid scope, partial acceptance and malformed evidence cannot display a score', () => {
  const changes = [
    d => { d.schema = 'other'; }, d => { d.status = 'running'; },
    d => { d.task = 'other'; }, d => { d.training_seed = 2; },
    d => { d.num_action_samples = 1; }, d => { d.eval_seed = null; },
    d => { d.env_seed = 1; }, d => { d.episodes = 50; },
    d => { d.completed_episodes = 7999; }, d => { d.completed_episodes = 8001; },
    d => { d.completed_episodes = '8000'; }, d => { d.successes = null; },
    d => { d.successes = 8001; }, d => { d.successes = -1; },
    d => { d.successes = 0.5; }, d => { delete d.receipt; },
    d => { d.receipt.sha256 = 'missing'; }, d => { d.receipt.bytes = 0; },
    d => { d.receipt.path = ''; }, d => { d.run_id = '<foreign>'; },
  ];
  for (const change of changes) {
    const p = page(), value = fixture();
    value.local_divl.full_grid_evaluation = fullGrid('accepted_complete');
    change(value.local_divl.full_grid_evaluation);
    assert.throws(() => p.render(value));
    assert.equal(p.elements.get('mulligan-local-divl-full-grid').hidden, true);
    assert.equal(p.elements.get('mulligan-local-divl-full-grid-results').hidden, true);
  }
  for (const change of [d => { d.successes = 1; }, d => { d.receipt = fullGrid('accepted_complete').receipt; }]) {
    const p = page(), value = fixture();
    value.local_divl.full_grid_evaluation = fullGrid();
    change(value.local_divl.full_grid_evaluation);
    assert.throws(() => p.render(value));
    assert.equal(p.elements.get('mulligan-local-divl-full-grid-results').hidden, true);
  }
});

test('full-grid template labels one seed, separate cohorts and no new video claim', () => {
  const section = template.match(/<div id="mulligan-local-divl-full-grid"[\s\S]*?<\/section>/)[0];
  assert.match(section, /50-start pilot and 400-episode training monitors/);
  assert.match(section, /one training seed/);
  assert.match(section, /does not establish a five-seed aggregate or paper match/);
  assert.match(section, /No new video is claimed/);
  assert.match(section, /Training seed 1 · 32 action candidates · evaluation seed 42 · environment seed 42/);
});
