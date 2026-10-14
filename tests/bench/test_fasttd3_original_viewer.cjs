// Explicit synthetic CPU DOM/media fixtures; no native results or browser proof.
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
  const element = tag => ({tag, children: [], dataset: {}, hidden: false, paused: false,
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; },
    setAttribute(name, value) { this[name] = String(value); },
    pause() { this.paused = true; },
    querySelectorAll(name) { return this.children.flatMap(node => [
      ...(node.tag === name ? [node] : []), ...(node.querySelectorAll ? node.querySelectorAll(name) : []),
    ]); }});
  const context = {document: {
    getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); },
    createElement: element,
    createElementNS(ns, tag) { assert.equal(ns, 'http://www.w3.org/2000/svg'); return element(tag); },
  }};
  vm.createContext(context);
  vm.runInContext(script, context);
  return {context, elements, render: value => context.renderFastOriginal(value)};
}

function fixture() {
  return {schema: 'fasttd3.original-hurdle/v1', status: 'accepted_complete_run',
    task: 'h1hand-hurdle-v0', seed: 1, num_envs: 128, completed_vector_calls: 100000,
    simulator_warning_count: 0,
    author_commit: '229ed59bbf43ea2f7a2d5d90d1076314839944d7',
    evaluations: Array.from({length: 19}, (_, i) => ({vector_index: (i + 1) * 5000,
      mean_return: i - 3.25, mean_length: 100.125 + i})),
    clips: [5000, 50000, 95000].map((vector_index, i) => {
      const sha256 = String(i + 1).padEnd(64, '0');
      return {vector_index, status: 'accepted_original_recording', sha256,
        video: sha256.slice(0, 16) + '.mp4', source_gif_sha256: 'a'.repeat(64),
        playback_timing: 'source_gif_unchanged'};
    })};
}

test('absent optional field stays hidden and preserves all existing panels', () => {
  const p = page();
  p.elements.set('fast-score', {textContent: 'Existing ABC fixture'});
  p.elements.set('catcher', {hidden: false});
  p.render(undefined);
  assert.equal(p.elements.get('fast-original').hidden, true);
  assert.equal(p.elements.has('fast-original-status'), false);
  assert.equal(p.elements.has('fast-original-rows'), false);
  assert.equal(p.elements.get('fast-score').textContent, 'Existing ABC fixture');
  assert.equal(p.elements.get('catcher').hidden, false);
  assert.match(template, /renderFastOriginal\(data\.fasttd3_original\)/);
  assert.match(template, /<h2>Earlier ABC comparison<\/h2>/);
});

test('explicit pending shows no score, curve, table or video', () => {
  const p = page();
  p.render({schema: 'fasttd3.original-hurdle/v1', status: 'pending'});
  assert.equal(p.elements.get('fast-original').hidden, false);
  assert.equal(p.elements.get('fast-original-results').hidden, true);
  assert.match(p.elements.get('fast-original-status').textContent, /acceptance pending/);
  assert.equal(p.elements.has('fast-original-curve'), false);
  assert.equal(p.elements.has('fast-original-rows'), false);
  assert.equal(p.elements.get('fast-original-clips').children.length, 0);
});

test('complete synthetic cohort renders mean-return curve, accessible table and fixed recording choices', () => {
  const p = page();
  p.render(fixture());
  assert.equal(p.elements.get('fast-original').hidden, false);
  assert.equal(p.elements.get('fast-original-results').hidden, false);
  assert.match(p.elements.get('fast-original-status').textContent, /does not establish a per-world success rate/);
  assert.match(p.elements.get('fast-original-warning').textContent, /No simulator warning messages.*scanned stdout/);
  assert.match(p.elements.get('fast-original-warning').textContent, /does not prove clean physics/);
  const rows = p.elements.get('fast-original-rows').children;
  assert.equal(rows.length, 19);
  assert.equal(rows[0].children[0].scope, 'row');
  assert.equal(rows[0].children[1].textContent, '-3.250');
  assert.equal(rows[18].children[0].textContent, '95,000');
  const svg = p.elements.get('fast-original-curve').children[0];
  assert.equal(svg.role, 'img');
  assert.equal(svg.querySelectorAll('circle').length, 19);
  assert.equal(svg.querySelectorAll('polyline')[0].points.split(' ').length, 19);
  assert.match(svg.querySelectorAll('desc')[0].textContent, /No evaluation.*100000/);
  const box = p.elements.get('fast-original-clips');
  const select = box.querySelectorAll('select')[0];
  assert.deepEqual(Array.from(select.children, node => node.value), [5000, 50000, 95000]);
  assert.equal(box.children[0].tag, 'label');
  const video = box.querySelectorAll('video')[0];
  assert.equal(video.controls, true);
  assert.equal(video.preload, 'metadata');
  assert.match(video['aria-label'], /H1Hand Hurdle.*5000/);
  assert.match(template, /not per-control traces or held-out episode evaluations/);
  assert.match(template, /final weights.*100,000; the original recipe does not evaluate/);
});

test('logged simulator warnings remain visible without hiding accepted return points or recordings', () => {
  const p = page(), data = fixture();
  data.simulator_warning_count = 8;
  p.render(data);
  assert.equal(p.elements.get('fast-original').hidden, false);
  assert.equal(p.elements.get('fast-original-results').hidden, false);
  const warning = p.elements.get('fast-original-warning').textContent;
  assert.match(warning, /completed with 8 logged simulator warnings/);
  assert.match(warning, /internal physics resets/);
  assert.match(warning, /per-world effects are unmeasured/);
  assert.match(warning, /printed messages in scanned stdout, not worlds or recovery events/);
  assert.equal(p.elements.get('fast-original-rows').children.length, 19);
  const box = p.elements.get('fast-original-clips');
  const select = box.querySelectorAll('select')[0];
  assert.deepEqual(Array.from(select.children, node => node.value), [5000, 50000, 95000]);
  assert.equal(box.querySelectorAll('video').length, 1);
  p.render({schema: 'fasttd3.original-hurdle/v1', status: 'pending'});
  assert.equal(p.elements.get('fast-original-warning').textContent, '');
});

test('missing or invalid simulator warning counts reject accepted records and clear prior media', () => {
  for (const value of [undefined, null, false, '8', -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    const p = page(), data = fixture();
    p.render(fixture());
    const video = p.elements.get('fast-original-clips').querySelectorAll('video')[0];
    if (value === undefined) delete data.simulator_warning_count;
    else data.simulator_warning_count = value;
    assert.throws(() => p.render(data), /nonnegative integer simulator warning count/);
    assert.equal(p.elements.get('fast-original').hidden, true);
    assert.equal(p.elements.get('fast-original-clips').children.length, 0);
    assert.equal(video.paused, true);
  }
});

test('selection pauses the prior clip; missing recording stays pending without media', () => {
  const p = page(), data = fixture();
  data.clips[1] = {vector_index: 50000, status: 'pending'};
  p.render(data);
  const box = p.elements.get('fast-original-clips');
  const first = box.querySelectorAll('video')[0], select = box.querySelectorAll('select')[0];
  select.value = '50000';
  select.onchange();
  assert.equal(first.paused, true);
  assert.equal(box.querySelectorAll('video').length, 0);
  assert.match(box.children[1].children[0].textContent, /recording pending; no media/);
  select.value = '95000';
  select.onchange();
  assert.match(box.querySelectorAll('video')[0]['aria-label'], /95000/);
});

test('unchanged refresh preserves selection, media identity and playback position', () => {
  const p = page();
  p.render(fixture());
  const box = p.elements.get('fast-original-clips');
  const select = box.querySelectorAll('select')[0];
  select.value = '95000';
  select.onchange();
  const video = box.querySelectorAll('video')[0];
  video.currentTime = 0.5;
  p.render(fixture());
  assert.equal(box.querySelectorAll('select')[0], select);
  assert.equal(box.querySelectorAll('video')[0], video);
  assert.equal(video.currentTime, 0.5);
  assert.equal(video.paused, false);
});

test('absent or pending refresh pauses and removes previously accepted media', () => {
  for (const value of [undefined, {schema: 'fasttd3.original-hurdle/v1', status: 'pending'}]) {
    const p = page();
    p.render(fixture());
    const box = p.elements.get('fast-original-clips'), video = box.querySelectorAll('video')[0];
    p.render(value);
    assert.equal(video.paused, true);
    assert.equal(box.querySelectorAll('video').length, 0);
  }
});

test('partial, foreign, malformed and retimed inputs fail without exposed results', () => {
  const changes = [
    d => { d.schema = 'other'; }, d => { d.status = 'running'; },
    d => { d.task = 'ABC'; }, d => { d.seed = 2; }, d => { d.num_envs = 64; },
    d => { d.completed_vector_calls = 95000; }, d => { d.author_commit = 'other'; },
    d => { d.evaluations.pop(); }, d => { d.evaluations[0].vector_index = 0; },
    d => { d.evaluations[18].vector_index = 100000; }, d => { d.evaluations[0].mean_return = NaN; },
    d => { d.evaluations[0].mean_return = '7.4'; }, d => { d.evaluations[0].mean_length = 1001; },
    d => { d.clips.pop(); }, d => { d.clips[0].vector_index = 100000; },
    d => { d.clips[0].video = '../foreign.mp4'; }, d => { d.clips[0].sha256 = 'missing'; },
    d => { d.clips[0].source_gif_sha256 = 'missing'; }, d => { d.clips[0].playback_timing = 'retimed'; },
    d => { d.clips[0].status = 'pending'; }, d => { d.status = 'pending'; },
  ];
  for (const change of changes) {
    const p = page(), data = fixture();
    p.render(fixture());
    const video = p.elements.get('fast-original-clips').querySelectorAll('video')[0];
    change(data);
    assert.throws(() => p.render(data));
    assert.equal(p.elements.get('fast-original').hidden, true);
    assert.equal(p.elements.get('fast-original-clips').children.length, 0);
    assert.equal(video.paused, true);
  }
});
