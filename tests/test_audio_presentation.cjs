const test = require('node:test');
const assert = require('node:assert/strict');
const { prepare } = require('../ui/static/audio-presentation.js');
const fs = require('node:fs');
const vm = require('node:vm');

function aasist(verdict = 'real') {
  return {
    status: 'ok', verdict, model_id: 'aasist', calibration_version: '1.0.0',
    calibrated_fake_probability: 0.2, score_orientation: 'higher_is_bonafide',
    threshold: 0, duration_seconds: 6, analysis_window_seconds: 4.0375, hop_seconds: 2,
    windows: [
      { start_seconds: 0, end_seconds: 4.0375, bonafide_score: -1.2, fake_score: 18, display_fake_score: 1, is_fake: true },
      { start_seconds: 2, end_seconds: 6, bonafide_score: 0.3, fake_score: 20, display_fake_score: 1, is_fake: false },
    ],
  };
}

test('calibrated real recording retains raw window flags without 100% fake claims', () => {
  const input = aasist();
  const before = JSON.stringify(input);
  const view = prepare(input);
  assert.equal(view.label, 'real');
  assert.equal(view.flagged, 1);
  assert.deepEqual(view.all_windows.map(w => w.fake_score), [-1.2, 0.3]);
  assert.deepEqual(view.all_windows.map(w => w.is_fake), [true, false]);
  assert.equal(view.formatScore(-1.2), '-1.200');
  assert.equal(view.top_windows[0].start, 0); // Order by bonafide score, not logit difference.
  assert.match(view.verdict_detail, /window flags use the raw window threshold/);
  assert.ok(view.min < -1.2 && view.max > 0.3);
  assert.equal(JSON.stringify(input), before);
});

test('uncertain verdict is preserved explicitly', () => {
  assert.equal(prepare(aasist('uncertain')).label, 'uncertain');
});

test('error or unknown verdict cannot become real', () => {
  assert.throws(() => prepare({ status: 'error', verdict: 'error', warnings: ['Empty audio'] }), /Empty audio/);
  assert.throws(() => prepare(aasist('unrecognized')), /No valid audio verdict/);
});

test('legacy windows use checkpoint threshold rather than hardcoded 0.50', () => {
  const view = prepare({
    label: 'real', threshold: 0.51, analyzed_seconds: 3,
    all_windows: [{ start: 0, end: 3, fake_score: 0.505 }],
  });
  assert.equal(view.flagged, 0);
  assert.equal(view.threshold, 0.51);
  assert.equal(view.formatScore(0.505), '0.5050');
  assert.equal(view.raw, false);
});

test('backend window flag takes precedence over rounded raw score', () => {
  const input = aasist();
  input.windows[0].bonafide_score = 0; // Rounded from a small negative number.
  assert.equal(prepare(input).all_windows[0].is_fake, true);
});

test('missing raw score is reported instead of falling back to a softmax percentage', () => {
  const input = aasist();
  delete input.windows[0].bonafide_score;
  assert.throws(() => prepare(input), /missing or invalid/);
});

test('all-zero windows retain a nonzero chart range', () => {
  const input = aasist();
  input.windows.forEach(w => { w.bonafide_score = 0; });
  const view = prepare(input);
  assert.ok(view.min < 0 && view.max > 0);
});

// Exercise the real upload response handler and renderer with a minimal DOM/canvas.
for (const verdict of ['real', 'uncertain', 'fake', 'error']) {
  test(`upload result renders ${verdict} without invented window probabilities`, async () => {
    const nodes = new Map();
    const axisLabels = [];
    const context = new Proxy({}, {
      get: (_, key) => key === 'fillText' ? (text) => axisLabels.push(text) : () => {},
    });
    function element() {
      return {
        textContent: '', innerHTML: '', className: '', children: [], handlers: {},
        files: [{}], currentTime: 0,
        classList: { add() {}, remove() {} },
        addEventListener(type, handler) { this.handlers[type] = handler; },
        setAttribute() {}, scrollIntoView() {}, load() {},
        querySelectorAll() { return []; },
        appendChild(child) { this.children.push(child); },
        getContext() { return context; },
        getBoundingClientRect() { return { width: 1000, left: 0 }; },
      };
    }
    const document = {
      getElementById(id) { if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id); },
      querySelectorAll() { return []; }, createElement: element, addEventListener() {},
    };
    const result = aasist(verdict);
    if (verdict === 'error') result.status = 'error';
    const sandbox = {
      document, window: { AudioPresentation: { prepare }, devicePixelRatio: 1 },
      FormData: class { append() {} },
      fetch: async () => ({ json: async () => ({ ok: true, filename: 'sample.wav', audio_url: '/sample', result }) }),
    };
    vm.runInNewContext(fs.readFileSync(require.resolve('../ui/static/app.js'), 'utf8'), sandbox);
    await nodes.get('audio-upload-form').handlers.submit({ preventDefault() {} });
    if (verdict === 'error') {
      assert.match(nodes.get('audio-error-msg').textContent, /No valid audio verdict/);
      assert.equal(nodes.get('audio-verdict-banner')?.className || '', '');
      return;
    }
    const banner = nodes.get('audio-verdict-banner');
    assert.match(banner.className, new RegExp(verdict === 'uncertain' ? 'verdict-uncertain' : verdict === 'real' ? 'verdict-safe' : 'verdict-danger'));
    assert.equal(nodes.get('audio-spliced-badge').textContent, '1/2 WINDOWS FLAGGED');
    const rows = nodes.get('top-windows-tbody').children.map(row => row.innerHTML).join('');
    assert.match(rows, /-1\.200/);
    assert.doesNotMatch(rows, /100\.0%|High Confidence|Manipulated Speech/);
    assert.ok(axisLabels.includes('Window threshold (0.000)'));
    assert.ok(axisLabels.every(text => !String(text).includes('%')));
    assert.equal(nodes.get('btn-submit-audio').disabled, false);
  });
}
