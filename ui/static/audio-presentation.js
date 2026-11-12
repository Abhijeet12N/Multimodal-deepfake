/* Display audio decisions without changing model scores or calibrating windows. */
(function (root) {
  'use strict';

  function prepare(result) {
    const label = result.verdict ?? result.label;
    if (result.status === 'error' || !['real', 'fake', 'uncertain'].includes(label)) {
      throw new Error((result.warnings || []).join(' ') || 'No valid audio verdict was returned.');
    }
    const service = Boolean(result.verdict);
    const raw = result.score_orientation === 'higher_is_bonafide';
    const calibrated = Boolean(result.calibration_version);
    const windowThreshold = result.window_threshold ?? result.threshold;
    const threshold = Number.isFinite(windowThreshold) ? windowThreshold : null;
    const source = service ? result.windows : result.all_windows;
    const windows = (source || []).map((win) => {
      const score = raw ? win.bonafide_score : win.fake_score;
      if (!Number.isFinite(score)) throw new Error('Audio window score is missing or invalid.');
      return {
        start: win.start_seconds ?? win.start,
        end: win.end_seconds ?? win.end,
        fake_score: score, // Chart value; raw bonafide score for AASIST.
        is_fake: typeof win.is_fake === 'boolean' ? win.is_fake
          : threshold === null ? null : raw ? score < threshold : score >= threshold,
      };
    });
    const flagged = windows.filter((win) => win.is_fake === true).length;
    const ranked = [...windows].sort((a, b) => raw
      ? a.fake_score - b.fake_score : b.fake_score - a.fake_score);
    const values = windows.map((win) => win.fake_score);
    let min = 0;
    let max = 1;
    if (raw) {
      min = Math.min(0, threshold ?? 0);
      max = Math.max(0, threshold ?? 0);
      values.forEach((score) => { min = Math.min(min, score); max = Math.max(max, score); });
      const pad = Math.max((max - min) * 0.1, 0.1);
      min -= pad;
      max += pad;
    }
    const formatScore = (score) => raw ? score.toFixed(3) : score.toFixed(4);
    const recordingScore = service ? result.raw_recording_score : result.fake_probability;
    const decisionScore = calibrated ? result.calibrated_fake_probability : recordingScore;
    const decisionThreshold = result.threshold;
    const recordingMetrics = [
      { label: 'Recording score (average)', value: Number.isFinite(recordingScore) ? formatScore(recordingScore) : 'Unavailable' },
    ];
    if (calibrated) {
      recordingMetrics.push({ label: 'Calibrated spoof score', value: Number.isFinite(decisionScore) ? decisionScore.toFixed(4) : 'Unavailable' });
    }
    recordingMetrics.push({ label: 'Decision threshold', value: Number.isFinite(decisionThreshold)
      ? calibrated ? decisionThreshold.toFixed(4) : formatScore(decisionThreshold) : 'Unavailable' });
    let detail = service
      ? `${result.model_id || 'Audio detector'}: ${windows.length} evaluated windows.`
      : `Wav2Vec 2.0: ${windows.length} evaluated windows.`;
    if (calibrated) {
      detail += ' Recording verdict uses calibration of the aggregate score; window flags use the raw window threshold.';
    } else {
      detail += ' Recording verdict uses the average window score and decision threshold.';
      if (!service && label === 'uncertain') {
        detail += ' The average is below the threshold, but localized window flags make this result inconclusive.';
      }
    }
    detail += ` ${flagged} of ${windows.length} windows crossed the diagnostic window threshold. The flagged count is not a majority vote or a percentage of fake audio. Window flags are diagnostics, not proof of manipulation.`;
    if (result.analysis_window_seconds != null) {
      detail += ` Window: ${result.analysis_window_seconds}s; hop: ${result.hop_seconds}s.`;
    }
    if (result.warnings?.length) detail += ` ${result.warnings.join(' ')}`;
    return {
      label,
      analyzed_seconds: result.duration_seconds ?? result.analyzed_seconds ?? 0,
      windows_analyzed: windows.length,
      all_windows: windows,
      top_windows: ranked.slice(0, 5),
      raw, threshold, min, max, formatScore, flagged,
      recording_metrics: recordingMetrics,
      prediction_note: calibrated
        ? 'Calibrated model prediction. This recording\'s authenticity has not been independently verified.'
        : 'Uncalibrated model prediction. This recording\'s authenticity has not been independently verified.',
      verdict_detail: detail,
      chart_title: raw ? 'Raw bonafide score vs time (higher = more bona fide)'
      : 'Window model score vs time (higher = more spoof-like)',
      score_label: raw ? 'Raw bonafide score' : 'Model score (0 to 1)',
      chart_note: raw
        ? 'Raw scores are not probabilities. Amber marks backend window flags; these can differ from the recording verdict. Click a bar to play that segment.'
        : 'Model scores are not validated window confidence percentages. Amber marks window flags. Click a bar to play that segment.',
    };
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { prepare };
  else root.AudioPresentation = { prepare };
})(typeof window !== 'undefined' ? window : globalThis);
