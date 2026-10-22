// Demo client logic
(function() {
    const $ = (id) => document.getElementById(id);
    const resultSection = $('result-section');
    const loader = $('loader');
    const resultBody = $('result-body');
    const errorBox = $('result-error');
    let lastReport = null;

    function showLoading() {
        resultSection.classList.remove('result-hidden');
        loader.classList.remove('hidden');
        resultBody.classList.add('hidden');
        errorBox.classList.add('hidden');
        // scroll into view
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function showError(msg) {
        loader.classList.add('hidden');
        resultBody.classList.add('hidden');
        errorBox.classList.remove('hidden');
        errorBox.textContent = '⚠️  ' + msg;
    }

    function severityFor(score) {
        if (score >= 0.85) return 'high';
        if (score >= 0.5)  return 'mid';
        return 'low';
    }

    function showResults(data) {
        loader.classList.add('hidden');
        errorBox.classList.add('hidden');
        resultBody.classList.remove('hidden');

        // Video
        const v = $('result-video');
        v.src = data.video_url;
        v.load();

        // Decision box
        const decBox = $('decision-box');
        decBox.classList.remove('severity-safe', 'severity-danger');
        decBox.classList.add('severity-' + data.decision.severity);
        $('decision-label').textContent = data.decision.label;
        $('decision-sub').textContent = data.decision.sub;

        // Meta — ground truth if known
        let metaText = '';
        if (data.ground_truth && data.ground_truth.category) {
            const gt = data.ground_truth;
            const dt = data.decision.label.includes('FAKE') === gt.any_fake;
            const tag = dt ? '✓ correct' : '✗ incorrect';
            metaText = `Ground truth: ${gt.category} (any_fake=${gt.any_fake ? 'yes' : 'no'}) · ${tag}`;
        } else {
            metaText = 'Ground truth: unknown (no category could be inferred from the filename)';
        }
        $('decision-meta').textContent = metaText;

        // 3 head scores with animated bars
        ['video', 'audio', 'any'].forEach((task, i) => {
            const sc = data.scores[task];
            const pct = Math.round(sc * 1000) / 10;
            const bar = $('bar-' + task);
            bar.style.width = '0%';
            bar.classList.remove('high', 'mid', 'low');
            bar.classList.add(severityFor(sc));
            // Animate in
            setTimeout(() => { bar.style.width = pct + '%'; }, 80 * (i + 1));

            $('score-' + task).textContent = sc.toFixed(4);

            // Truth match indicator
            const tEl = $('truth-' + task);
            if (data.ground_truth && data.correct) {
                const gtKey = task === 'any' ? 'any_fake' : (task + '_fake');
                const gt = data.ground_truth[gtKey];
                const ok = data.correct[task];
                tEl.textContent = `Actual: ${gt ? 'fake' : 'real'} · ${ok ? '✓' : '✗'}`;
                tEl.className = 'head-truth ' + (ok ? 'ok' : 'bad');
            } else {
                tEl.textContent = '';
                tEl.className = 'head-truth';
            }
        });

        // Info row
        const info = data.info || {};
        $('info-face').textContent  = info.face_prob != null ? info.face_prob.toFixed(3) : '—';
        $('info-rms').textContent   = info.audio_rms != null ? info.audio_rms.toFixed(4) : '—';
        $('info-pre').textContent   = (info.preprocess_s ?? '—') + ' s';
        $('info-inf').textContent   = (info.inference_s  ?? '—') + ' s';
        $('info-total').textContent = (data.elapsed_total ?? '—') + ' s';

        // Structured explanation report
        lastReport = data.report || null;
        $('report-summary').textContent = lastReport ? lastReport.summary : 'Could not generate the report.';
        const findings = $('report-findings');
        findings.innerHTML = '';
        (lastReport?.findings || []).forEach((finding) => {
            const item = document.createElement('li');
            item.textContent = finding;
            findings.appendChild(item);
        });
    }

    async function analyze(payload) {
        showLoading();
        try {
            const res = await fetch('/api/analyze', { method: 'POST', body: payload });
            const data = await res.json();
            if (!data.ok) {
                showError(data.error || 'Unknown error');
                return;
            }
            showResults(data);
        } catch (e) {
            showError('Request error: ' + e.message);
        }
    }

    // Sample card click
    document.querySelectorAll('.sample-card').forEach(btn => {
        btn.addEventListener('click', () => {
            const fd = new FormData();
            fd.append('mode', 'sample');
            fd.append('sample', btn.dataset.sample);
            analyze(fd);
        });
    });

    // Upload form
    const form = $('upload-form');
    if (form) {
        form.addEventListener('submit', (ev) => {
            ev.preventDefault();
            const fileInput = $('file-input');
            if (!fileInput.files.length) {
                showError('Choose a video first.');
                return;
            }
            const fd = new FormData();
            fd.append('mode', 'upload');
            fd.append('file', fileInput.files[0]);
            analyze(fd);
        });
    }

    // Standalone audio checker. This uses a different endpoint/checkpoint from video analysis.
    const audioForm = $('audio-upload-form');
    if (audioForm) {
        audioForm.addEventListener('submit', async (ev) => {
            ev.preventDefault();
            const input = $('audio-file-input');
            const result = $('audio-result');
            if (!input.files.length) {
                result.classList.remove('hidden', 'audio-ok', 'audio-error');
                result.classList.add('audio-error');
                result.textContent = 'Choose an audio file first.';
                return;
            }
            result.classList.remove('hidden', 'audio-ok', 'audio-error');
            result.textContent = 'Analyzing audio…';
            const fd = new FormData();
            fd.append('file', input.files[0]);
            try {
                const response = await fetch('/api/analyze-audio', { method: 'POST', body: fd });
                const data = await response.json();
                result.classList.remove('audio-ok', 'audio-error');
                if (!data.ok) {
                    result.classList.add('audio-error');
                    result.textContent = data.error || 'Audio analysis failed.';
                    return;
                }
                const isFake = data.result.label === 'fake';
                result.classList.add(isFake ? 'audio-error' : 'audio-ok');
                const strongest = (data.result.top_windows || [])
                    .slice(0, 3)
                    .map((window) => `${window.start_seconds}-${window.end_seconds}s (${(window.fake_score * 100).toFixed(0)}%)`)
                    .join(', ');
                const labelTag = isFake ? (data.result.is_spliced ? 'FAKE audio (Spliced / Partial Synthetic)' : 'FAKE audio') : 'REAL audio';
                const peakInfo = data.result.peak_probability ? ` · peak window: ${(data.result.peak_probability * 100).toFixed(1)}%` : '';
                result.textContent = `${data.filename}: ${labelTag} · whole-file fake score ${(data.result.fake_probability * 100).toFixed(1)}%${peakInfo} · ${data.result.windows_analyzed} windows across ${data.result.analyzed_seconds}s · strongest windows: ${strongest} · ${data.result.inference_seconds}s`;
            } catch (error) {
                result.classList.add('audio-error');
                result.textContent = 'Request failed: ' + error.message;
            }
        });
    }

    const downloadBtn = $('download-report');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            if (!lastReport) return;
            const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'deepfake-analysis-report.json';
            link.click();
            URL.revokeObjectURL(url);
        });
    }
})();
