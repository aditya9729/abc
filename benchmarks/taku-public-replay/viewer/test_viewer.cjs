// Explicit display fixtures. No browser, accepted native media or policy execution.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/refreshMedia\(\);\s*$/, '');
const keys = ['robot_log_replay', 'reconstructed_laundry_scene'];

function page() {
  const elements = new Map();
  const element = tag => ({tag, children: [], dataset: {}, hidden: true,
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; },
    setAttribute(name, value) { this[name] = value; },
    pause() { this.paused = true; },
    querySelectorAll(name) { return this.children.filter(node => node.tag === name); }});
  const context = {document: {
    getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); },
    createElement: element,
  }};
  vm.createContext(context);
  vm.runInContext(script, context);
  return {elements, render: value => context.renderMedia(value)};
}

function fixture(accepted = false) {
  const value = {schema: 'taku.local-scene-viewer/v1'};
  keys.forEach((key, index) => {
    const sha256 = String(index + 1).repeat(64);
    value[key] = accepted ? {status: 'accepted_complete', video: sha256.slice(0, 16) + '.mp4',
      sha256, frames: 869, fps: 12, width: 640, height: 480,
      receipt: {path: '/explicit/synthetic/acceptance.json', bytes: 1, sha256: 'a'.repeat(64)}} : {status: 'pending'};
  });
  return value;
}

test('prepared manifest has two pending cards and no actual video or score', () => {
  const value = JSON.parse(fs.readFileSync(path.join(__dirname, 'media.json'), 'utf8'));
  const p = page();
  p.render(value);
  keys.forEach(key => {
    assert.equal(value[key].status, 'pending');
    assert.equal(p.elements.get(key + '-player').hidden, true);
    assert.equal(p.elements.get(key + '-player').children.length, 0);
    assert.match(p.elements.get(key + '-status').textContent, /Pending: .* video is not available/);
  });
});

test('accepted synthetic full media creates separate native-control players', () => {
  const p = page(), value = fixture(true);
  p.render(value);
  keys.forEach(key => {
    const player = p.elements.get(key + '-player'), video = player.children[0];
    assert.equal(player.hidden, false);
    assert.equal(video.controls, true);
    assert.equal(video.playsInline, true);
    assert.equal(video.src, value[key].video);
    assert.equal(p.elements.get(key + '-status').textContent, 'Complete recording · 72.42 seconds · 12 fps.');
  });
});

test('one pending reconstruction never borrows the accepted robot-only video', () => {
  const p = page(), value = fixture(true);
  value.reconstructed_laundry_scene = {status: 'pending'};
  p.render(value);
  assert.equal(p.elements.get('robot_log_replay-player').children.length, 2);
  assert.equal(p.elements.get('reconstructed_laundry_scene-player').children.length, 0);
});

test('unchanged refresh preserves players and positions; pending removes and pauses them', () => {
  const p = page(), value = fixture(true);
  p.render(value);
  const videos = keys.map(key => p.elements.get(key + '-player').children[0]);
  videos.forEach(video => { video.currentTime = 32; });
  p.render(structuredClone(value));
  keys.forEach((key, index) => {
    assert.equal(p.elements.get(key + '-player').children[0], videos[index]);
    assert.equal(videos[index].currentTime, 32);
  });
  p.render(fixture());
  videos.forEach(video => assert.equal(video.paused, true));
});

test('unaccepted, shortened or unsafe media rejects before either player changes', () => {
  const changes = [
    e => { e.status = 'rendering'; }, e => { e.video = '../video.mp4'; },
    e => { e.video = 'https://example.invalid/video.mp4'; }, e => { e.frames = 868; },
    e => { e.fps = 24; }, e => { e.sha256 = 'missing'; },
    e => { e.width = 0; }, e => { e.height = 0; }, e => { delete e.receipt; },
    e => { e.receipt.sha256 = ''; }, e => { e.receipt.bytes = 0; },
  ];
  for (const change of changes) {
    const p = page(), value = fixture(true);
    change(value.reconstructed_laundry_scene);
    assert.throws(() => p.render(value));
    assert.equal(p.elements.has('robot_log_replay-player'), false);
  }
  const p = page(), pending = fixture();
  pending.robot_log_replay.video = 'a'.repeat(16) + '.mp4';
  assert.throws(() => p.render(pending));
});

test('reference and fidelity labels distinguish the author image from local reconstruction', () => {
  assert.match(html, /src="source-reference.jpg"/);
  assert.match(html, /overlay describes the author's video/);
  assert.match(html, /not synchronized with the public 72-second log/);
  assert.match(html, /Recorded motion; estimated props; static towels; no learned controller/);
  assert.match(html, /No released Taku policy runs here/);
  assert.match(html, /reports no task score/);
  assert.ok(html.indexOf('id="scene-heading"') < html.indexOf('id="replay-heading"'));
  assert.match(html, /<details>\s*<summary id="fidelity-heading">/);
  assert.equal(/<script[^>]+src=/.test(html), false);
});
