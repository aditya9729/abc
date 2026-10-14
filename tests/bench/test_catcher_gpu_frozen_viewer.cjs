// Explicit CPU DOM and media fixtures. No native results, videos, or browser proof.
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
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    setAttribute(name, value) { this[name] = value; }});
  const context = {document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element(id));
      return elements.get(id);
    },
    createElement: element,
  }};
  vm.createContext(context);
  vm.runInContext(script, context);
  return {context, elements, render: value => context.renderCatcherGpuFrozen(value)};
}

function fixture() {
  // All counts, hashes, outcomes and paths below are explicitly synthetic.
  return {schema: 'catcher.gpu-frozen-comparison/v1', status: 'accepted_matched_development',
    restore: {status: 'original_full_checkpoint_restore_verified', checkpoint_step: 4096,
      original_update_calls: 1234,
      counters: {actor_train_state: {step: 1234}, critic: {step: 24680}, filter_critic: {step: 24680}},
      receipt: {path: '/home/user/aditya/explicit-synthetic-fixture/restore.json', bytes: 123, sha256: 'b'.repeat(64)}},
    rows: [100001, 100002, 100003, 100004].map((seed, i) => {
      const clip = (arm, solved) => {
        const sha256 = (i * 2 + arm + 1).toString(16).padEnd(64, '0');
        return {status: 'accepted_frozen_case', complete: true, solved, controls: 20 + i,
          frames: 21 + i, sha256, video: sha256.slice(0, 16) + '.mp4'};
      };
      return {seed, rtc_bc24: clip(0, i < 2), saved_gpu4096: clip(1, i === 0)};
    })};
}

test('absent comparison hides GPU panel and supplies no default score or update count', () => {
  const p = page();
  p.render(undefined);
  assert.equal(p.elements.get('catcher-gpu-frozen').hidden, true);
  assert.equal(p.elements.has('catcher-gpu-frozen-summary'), false);
  assert.equal(p.elements.has('catcher-gpu-frozen-updates'), false);
  assert.equal(p.elements.has('catcher-gpu-frozen-rows'), false);
});

test('synthetic accepted record renders exact cohort, eight full videos and actual restore clocks', () => {
  const p = page();
  p.render(fixture());
  assert.equal(p.elements.get('catcher-gpu-frozen').hidden, false);
  assert.match(p.elements.get('catcher-gpu-frozen-summary').textContent, /BC24 2\/4 · learned GPU checkpoint 1\/4/);
  assert.match(p.elements.get('catcher-gpu-frozen-updates').textContent, /index 4,096 · actor updates 1,234 · critic updates 24,680 · filter updates 24,680/);
  const rows = p.elements.get('catcher-gpu-frozen-rows').children;
  assert.equal(rows.length, 4);
  for (let i = 0; i < rows.length; i++) {
    assert.equal(rows[i].children[0].textContent, 100001 + i);
    for (const cell of rows[i].children.slice(1)) {
      const video = cell.children[1];
      assert.equal(video.tag, 'video');
      assert.equal(video.controls, true);
      assert.equal(video.preload, 'metadata');
      assert.match(video['aria-label'], new RegExp('seed ' + (100001 + i)));
      assert.match(video.src, /^[0-9a-f]{16}\.mp4$/);
    }
  }
});

test('refresh preserves playing and paused video identity and seek position', () => {
  const p = page();
  p.render(fixture());
  const rows = p.elements.get('catcher-gpu-frozen-rows').children;
  const first = rows[0].children[1].children[1];
  const second = rows[0].children[2].children[1];
  first.currentTime = 2.25; first.paused = false;
  second.currentTime = 1.5; second.paused = true;
  p.render(fixture());
  assert.equal(p.elements.get('catcher-gpu-frozen-rows').children, rows);
  assert.equal(rows[0].children[1].children[1], first);
  assert.equal(first.currentTime, 2.25); assert.equal(first.paused, false);
  assert.equal(second.currentTime, 1.5); assert.equal(second.paused, true);
});

test('missing or inconsistent restore metadata never fabricates4096 updates', () => {
  const changes = [
    value => { delete value.restore; },
    value => { value.restore.status = 'pending'; },
    value => { value.restore.checkpoint_step = 3072; },
    value => { delete value.restore.original_update_calls; },
    value => { value.restore.original_update_calls = 0; },
    value => { value.restore.original_update_calls = '1234'; },
    value => { delete value.restore.receipt; },
    value => { value.restore.receipt.bytes = 0; },
    value => { value.restore.receipt.sha256 = 'missing'; },
    value => { value.restore.counters.actor_train_state.step = 4096; },
    value => { value.restore.counters.critic.step--; },
    value => { value.restore.counters.filter_critic.step--; },
  ];
  for (const change of changes) {
    const p = page(); const value = fixture(); change(value);
    assert.throws(() => p.render(value));
    assert.equal(p.elements.get('catcher-gpu-frozen').hidden, true);
    assert.equal(p.elements.has('catcher-gpu-frozen-updates'), false);
  }
});

test('incomplete cases, wrong seeds and unsafe media are rejected before a score is visible', () => {
  const changes = [
    value => { value.status = 'running'; },
    value => { value.schema = 'catcher.frozen-comparison/v1'; },
    value => { value.rows.pop(); },
    value => { value.rows[1].seed = 100001; },
    value => { delete value.rows[0].saved_gpu4096; },
    value => { value.rows[0].rtc_bc24.status = 'failed'; },
    value => { value.rows[0].rtc_bc24.solved = null; },
    value => { value.rows[0].rtc_bc24.complete = false; },
    value => { value.rows[0].rtc_bc24.controls = 257; },
    value => { value.rows[0].rtc_bc24.frames--; },
    value => { value.rows[0].rtc_bc24.video = '../untrusted.mp4'; },
    value => { value.rows[0].rtc_bc24.video = 'https://example.com/untrusted.mp4'; },
    value => { value.rows[0].rtc_bc24.video = 'aaaaaaaaaaaaaaaa.mp4'; },
  ];
  for (const change of changes) {
    const p = page(); const value = fixture(); change(value);
    assert.throws(() => p.render(value));
    assert.equal(p.elements.get('catcher-gpu-frozen').hidden, true);
    assert.equal(p.elements.has('catcher-gpu-frozen-summary'), false);
  }
});

test('parent dispatch preserves the older CPU-R2 comparison and its active players', () => {
  const p = page(); const value = fixture();
  const old = {schema: 'catcher.frozen-comparison/v1', status: 'accepted_matched_development',
    rows: value.rows.map(row => ({seed: row.seed, rtc_bc24: row.rtc_bc24, saved_r2: row.saved_gpu4096}))};
  const result = {learning_status: 'Explicit synthetic fixture only', frozen_comparison: old,
    profiles: [0, 1, 2].map(i => ({label: 'Synthetic ' + i, video: 'a'.repeat(16) + '.mp4', episode_length: 10, solved: false}))};
  p.context.renderCatcher(result);
  const oldRows = p.elements.get('catcher-frozen-rows').children;
  const oldVideo = oldRows[0].children[1].children[1];
  oldVideo.currentTime = 3.25; oldVideo.paused = false;
  assert.equal(p.elements.get('catcher-gpu-frozen').hidden, true);
  p.context.renderCatcher({...result, gpu_frozen_comparison: value});
  assert.equal(p.elements.get('catcher-frozen-rows').children, oldRows);
  assert.equal(oldRows[0].children[1].children[1], oldVideo);
  assert.equal(oldVideo.currentTime, 3.25); assert.equal(oldVideo.paused, false);
  assert.equal(p.elements.get('catcher-gpu-frozen').hidden, false);
  assert.match(p.elements.get('catcher-frozen-summary').textContent, /learned R2/);
  p.context.renderCatcher(result);
  assert.equal(p.elements.get('catcher-gpu-frozen').hidden, true);
  assert.equal(p.elements.get('catcher-frozen').hidden, false);
});
