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
  return {context, elements, render: value => context.renderCatcherFrozen(value)};
}

function fixture() {
  return {schema: 'catcher.frozen-comparison/v1', status: 'accepted_matched_development',
    rows: [100001, 100002, 100003, 100004].map((seed, i) => {
      const clip = (arm, solved) => {
        const sha256 = (i * 2 + arm + 1).toString(16).padEnd(64, '0');
        return {status: 'accepted_frozen_case', complete: true, solved, controls: 20 + i,
          frames: 21 + i, sha256, video: sha256.slice(0, 16) + '.mp4'};
      };
      return {seed, rtc_bc24: clip(0, i < 2), saved_r2: clip(1, i === 0)};
    })};
}

test('absent optional key hides the panel and creates no scores', () => {
  const p = page();
  p.render(undefined);
  assert.equal(p.elements.get('catcher-frozen').hidden, true);
  assert.equal(p.elements.has('catcher-frozen-summary'), false);
  assert.equal(p.elements.has('catcher-frozen-rows'), false);
});

test('accepted CPU fixture renders four paired rows and eight complete media controls', () => {
  const p = page();
  p.render(fixture());
  const rows = p.elements.get('catcher-frozen-rows').children;
  assert.equal(p.elements.get('catcher-frozen').hidden, false);
  assert.equal(rows.length, 4);
  assert.match(p.elements.get('catcher-frozen-summary').textContent, /BC24 2\/4 · learned R2 1\/4/);
  for (let i = 0; i < rows.length; i++) {
    assert.equal(rows[i].children[0].textContent, 100001 + i);
    for (const cell of rows[i].children.slice(1)) {
      const video = cell.children[1];
      assert.equal(video.tag, 'video');
      assert.equal(video.controls, true);
      assert.equal(video.preload, 'metadata');
      assert.match(video['aria-label'], new RegExp('seed ' + (100001 + i)));
    }
  }
});

test('unchanged refresh preserves video identity and playback position', () => {
  const p = page();
  p.render(fixture());
  const rows = p.elements.get('catcher-frozen-rows').children;
  const video = rows[0].children[1].children[1];
  video.currentTime = 0.25;
  p.render(fixture());
  assert.equal(p.elements.get('catcher-frozen-rows').children, rows);
  assert.equal(rows[0].children[1].children[1], video);
  assert.equal(video.currentTime, 0.25);
});

test('unaccepted, partial, duplicated or malformed outcomes remain hidden', () => {
  const changes = [
    value => { value.status = 'running'; },
    value => { value.rows.pop(); },
    value => { value.rows[1].seed = 100001; },
    value => { value.rows[0].rtc_bc24 = null; },
    value => { value.rows[0].rtc_bc24.status = 'pending'; },
    value => { value.rows[0].rtc_bc24.solved = null; },
    value => { value.rows[0].rtc_bc24.complete = false; },
    value => { value.rows[0].rtc_bc24.controls = 0; },
    value => { value.rows[0].rtc_bc24.controls = 257; },
    value => { value.rows[0].rtc_bc24.frames--; },
    value => { value.rows[0].rtc_bc24.video = '../untrusted.mp4'; },
    value => { value.rows[0].rtc_bc24.video = 'aaaaaaaaaaaaaaaa.mp4'; },
    value => { value.rows[0].rtc_bc24.sha256 = 'missing'; },
  ];
  for (const change of changes) {
    const p = page();
    const value = fixture();
    change(value);
    assert.throws(() => p.render(value));
    assert.equal(p.elements.get('catcher-frozen').hidden, true);
    assert.equal(p.elements.has('catcher-frozen-summary'), false);
  }
});

test('invalid refresh hides accepted fixture; absent comparison keeps existing Catcher panels', () => {
  const p = page();
  p.render(fixture());
  assert.throws(() => p.render({...fixture(), status: 'failed'}));
  assert.equal(p.elements.get('catcher-frozen').hidden, true);
  p.context.renderCatcher({learning_status: 'Explicit CPU fixture', profiles: [0, 1, 2].map(i => ({
    label: 'Fixture ' + i, video: 'a'.repeat(16) + '.mp4', episode_length: 10, solved: false,
  }))});
  assert.equal(p.elements.get('catcher-frozen').hidden, true);
  assert.equal(p.elements.get('catcher-viewer').children.length, 3);
  assert.equal(p.elements.get('catcher-learning').textContent, 'Explicit CPU fixture');
});
