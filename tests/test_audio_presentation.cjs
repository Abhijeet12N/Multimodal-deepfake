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

function podcastResult() {
  return {
    status: 'ok', verdict: 'real', model_id: 'legacy_wav2vec_mlp',
    raw_recording_score: 0.911344, threshold: 0.9849822521209717,
    window_threshold: 0.9849822521209717, calibration_version: null,
    score_orientation: 'higher_is_fake', duration_seconds: 68.976,
    analysis_window_seconds: 2, hop_seconds: 1,
    windows: Array.from({ length: 63 }, (_, i) => ({
      start_seconds: i, end_seconds: i + 2,
      fake_score: i < 45 ? 0.999 : 0.692204, is_fake: i < 45,
    })),
  };
}

test('majority of flagged windows does not override the recording mean or imply verified authenticity', () => {
  const input = podcastResult();
  const before = JSON.stringify(input);
  const view = prepare(input);
  assert.equal(view.label, 'real');
  assert.equal(view.flagged, 45);
  assert.deepEqual(view.recording_metrics, [
    { label: 'Recording score (average)', value: '0.9113' },
    { label: 'Decision threshold', value: '0.9850' },
  ]);
  assert.match(view.verdict_detail, /average window score/);
  assert.match(view.verdict_detail, /45 of 63 windows crossed/);
  assert.match(view.verdict_detail, /not a majority vote or a percentage of fake audio/);
  assert.match(view.prediction_note, /Uncalibrated model prediction/);
  assert.equal(JSON.stringify(input), before);
});

test('calibrated recording metrics keep raw average, calibrated score and decision threshold separate', () => {
  const input = { ...aasist(), raw_recording_score: -0.45, threshold: 0.5, window_threshold: 0 };
  const view = prepare(input);
  assert.deepEqual(view.recording_metrics, [
    { label: 'Recording score (average)', value: '-0.450' },
    { label: 'Calibrated spoof score', value: '0.2000' },
    { label: 'Decision threshold', value: '0.5000' },
  ]);
  assert.equal(view.threshold, 0);
  assert.match(view.prediction_note, /^Calibrated model prediction/);
});

test('standalone detector exposes its recording score and keeps its inconclusive outcome', () => {
  const view = prepare({
    label: 'uncertain', fake_probability: 0.47, threshold: 0.51,
    all_windows: [{ start: 0, end: 3, fake_score: 0.99 }],
  });
  assert.equal(view.recording_metrics[0].value, '0.4700');
  assert.equal(view.label, 'uncertain');
  assert.match(view.verdict_detail, /localized window flags make this result inconclusive/);
});

test('missing recording score is unavailable instead of invented from rounded windows', () => {
  assert.equal(prepare(aasist()).recording_metrics[0].value, 'Unavailable');
});

// Exercise the real upload response handler and renderer with a minimal DOM/canvas.
function mountAudioUI(result) {
    const nodes = new Map();
    const documentHandlers = {};
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
      querySelectorAll() { return []; }, createElement: element,
      addEventListener(type, handler) { documentHandlers[type] = handler; },
    };
    const sandbox = {
      document, window: { AudioPresentation: { prepare }, devicePixelRatio: 1, addEventListener() {} },
      FormData: class { append() {} },
      fetch: async () => ({ json: async () => ({ ok: true, filename: 'sample.wav', audio_url: '/sample', result }) }),
    };
    vm.runInNewContext(fs.readFileSync(require.resolve('../ui/static/app.js'), 'utf8'), sandbox);
    documentHandlers.DOMContentLoaded();
    return { nodes, axisLabels };
}

for (const verdict of ['real', 'uncertain', 'fake', 'error']) {
  test(`upload result renders ${verdict} without invented window probabilities`, async () => {
    const result = aasist(verdict);
    if (verdict === 'error') result.status = 'error';
    const { nodes, axisLabels } = mountAudioUI(result);
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

test('podcast upload renders recording score, threshold and prediction qualification beside verdict', async () => {
  const { nodes } = mountAudioUI(podcastResult());
  await nodes.get('audio-upload-form').handlers.submit({ preventDefault() {} });
  assert.equal(nodes.get('audio-verdict-title').textContent, 'Model Predicts: Bona Fide');
  assert.equal(nodes.get('audio-verdict-badge').textContent, 'MODEL PREDICTION: BONA FIDE');
  assert.equal(nodes.get('audio-spliced-badge').textContent, '45/63 WINDOWS FLAGGED');
  assert.match(nodes.get('audio-prediction-note').textContent, /Uncalibrated model prediction/);
  const metrics = nodes.get('audio-recording-metrics').children;
  assert.equal(metrics[0].children[1].textContent, '0.9113');
  assert.equal(metrics[1].children[1].textContent, '0.9850');
});

test('timer initializes before playback, follows metadata and seeking, and resets for another file', async () => {
  const result = podcastResult();
  const { nodes } = mountAudioUI(result);
  const submit = () => nodes.get('audio-upload-form').handlers.submit({ preventDefault() {} });
  await submit();
  const player = nodes.get('result-audio-player');
  const timer = nodes.get('playback-timer');
  assert.equal(timer.textContent, '0:00 / 1:08');

  player.duration = 69.1;
  player.handlers.loadedmetadata();
  assert.equal(timer.textContent, '0:00 / 1:09');
  player.duration = 70;
  player.handlers.durationchange();
  player.currentTime = 12.3;
  player.handlers.seeked();
  assert.equal(timer.textContent, '0:12 / 1:10');
  player.currentTime = 13.1;
  player.handlers.timeupdate();
  assert.equal(timer.textContent, '0:13 / 1:10');

  result.duration_seconds = 11.4;
  await submit();
  assert.equal(timer.textContent, '0:00 / 0:11');
  player.currentTime = 0;
  player.duration = NaN;
  player.handlers.emptied();
  assert.equal(timer.textContent, '0:00 / 0:11');
  player.duration = Infinity;
  player.currentTime = NaN;
  player.handlers.durationchange();
  assert.equal(timer.textContent, '0:00 / 0:11');
});
