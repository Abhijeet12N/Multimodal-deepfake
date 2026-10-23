/**
 * Deepfake Detection Studio — Forensic Client Application
 * Handles Tabs, Drag & Drop, Pipeline Stepper, Circular Gauges,
 * Inspection Gallery, and Interactive Audio Timeline Canvas.
 */

(function () {
  'use strict';

  // Helper shortcuts
  const $ = (id) => document.getElementById(id);
  const $$ = (sel) => document.querySelectorAll(sel);

  let lastVideoReport = null;
  let activeAudioData = null;
  let pipelineTimer = null;

  // ========================================================
  // 1. Tab Navigation
  // ========================================================
  function initTabs() {
    const tabBtns = $$('.tab-btn');
    const panels = $$('.tab-panel');

    tabBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.tab;

        tabBtns.forEach((b) => {
          b.classList.remove('active');
          b.setAttribute('aria-selected', 'false');
        });
        panels.forEach((p) => {
          p.classList.remove('active');
          p.classList.add('hidden');
        });

        btn.classList.add('active');
        btn.setAttribute('aria-selected', 'true');

        const activePanel = $(`panel-${target}`);
        if (activePanel) {
          activePanel.classList.remove('hidden');
          activePanel.classList.add('active');
        }

        // Trigger canvas resize if switching to audio tab
        if (target === 'audio' && activeAudioData) {
          requestAnimationFrame(() => renderAudioTimeline(activeAudioData));
        }
      });
    });
  }

  // ========================================================
  // 2. Drag & Drop Upload Zones with Instant Previews
  // ========================================================
  function setupDropzone({ dropzoneId, inputId, defaultContentId, previewWrapId, previewElId, clearBtnId, mediaType }) {
    const dropzone = $(dropzoneId);
    const input = $(inputId);
    const defaultContent = $(defaultContentId);
    const previewWrap = $(previewWrapId);
    const previewEl = $(previewElId);
    const clearBtn = $(clearBtnId);

    if (!dropzone || !input) return;

    ['dragenter', 'dragover'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        input.files = files;
        handlePreview(files[0]);
      }
    });

    input.addEventListener('change', () => {
      if (input.files.length > 0) {
        handlePreview(input.files[0]);
      }
    });

    if (clearBtn) {
      clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        input.value = '';
        if (previewEl) {
          previewEl.pause();
          previewEl.src = '';
        }
        previewWrap.classList.add('hidden');
        defaultContent.classList.remove('hidden');
      });
    }

    function handlePreview(file) {
      if (!file) return;
      const url = URL.createObjectURL(file);
      if (previewEl) {
        previewEl.src = url;
      }

      if (mediaType === 'video') {
        const fn = $('preview-filename');
        const fs = $('preview-filesize');
        if (fn) fn.textContent = file.name;
        if (fs) fs.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
      } else if (mediaType === 'audio') {
        const fn = $('audio-preview-filename');
        if (fn) fn.textContent = file.name;
      }

      defaultContent.classList.add('hidden');
      previewWrap.classList.remove('hidden');
    }
  }

  // ========================================================
  // 3. Pipeline Stepper Simulation
  // ========================================================
  function startPipelineStepper() {
    clearInterval(pipelineTimer);
    $('video-idle-card')?.classList.add('hidden');
    $('video-results-container')?.classList.add('hidden');
    $('video-error-box')?.classList.add('hidden');
    $('pipeline-stepper-card')?.classList.remove('hidden');

    const steps = [$('pstep-1'), $('pstep-2'), $('pstep-3'), $('pstep-4')];
    steps.forEach((step, idx) => {
      step.className = 'pipeline-step';
      step.querySelector('.step-icon').textContent = '○';
    });

    let currentStep = 0;
    function advance() {
      if (currentStep < steps.length) {
        if (currentStep > 0) {
          steps[currentStep - 1].className = 'pipeline-step step-done';
          steps[currentStep - 1].querySelector('.step-icon').textContent = '✓';
        }
        steps[currentStep].className = 'pipeline-step step-active';
        steps[currentStep].querySelector('.step-icon').innerHTML = '<span class="spinner-sm"></span>';
        currentStep++;
      }
    }

    advance();
    pipelineTimer = setInterval(advance, 1400);
  }

  function stopPipelineStepper() {
    clearInterval(pipelineTimer);
    $('pipeline-stepper-card')?.classList.add('hidden');
  }

  // ========================================================
  // 4. Semi-Radial Circular Gauge Animation
  // ========================================================
  const GAUGE_CIRCUMFERENCE = 141.37; // Arc circumference for r=45 semi-circle

  function setGauge(meterId, valId, scoreId, score) {
    const meter = $(meterId);
    const valText = $(valId);
    const scoreText = $(scoreId);
    if (!meter) return;

    const clamped = Math.max(0, Math.min(1, score));
    const pct = Math.round(clamped * 1000) / 10;
    const offset = GAUGE_CIRCUMFERENCE * (1 - clamped);

    // Color assignment
    let strokeColor = 'var(--emerald)';
    if (clamped >= 0.85) {
      strokeColor = 'var(--crimson)';
    } else if (clamped >= 0.50) {
      strokeColor = 'var(--amber)';
    }

    meter.style.stroke = strokeColor;
    meter.style.strokeDashoffset = offset;

    if (valText) {
      // Counter animation
      let start = 0;
      const duration = 600;
      const startTime = performance.now();
      function updateCounter(now) {
        const elapsed = now - startTime;
        const progress = Math.min(1, elapsed / duration);
        const curr = (progress * pct).toFixed(1);
        valText.textContent = curr + '%';
        if (progress < 1) requestAnimationFrame(updateCounter);
      }
      requestAnimationFrame(updateCounter);
    }

    if (scoreText) {
      scoreText.textContent = clamped.toFixed(4);
    }
  }

  // ========================================================
  // 5. Multimodal Video Results Renderer
  // ========================================================
  function showVideoResults(data) {
    stopPipelineStepper();
    $('video-error-box')?.classList.add('hidden');
    const resultsContainer = $('video-results-container');
    if (!resultsContainer) return;
    resultsContainer.classList.remove('hidden');

    // 1. Verdict Banner
    const banner = $('verdict-banner');
    const vBadge = $('verdict-badge');
    const rBadge = $('risk-badge');
    const vTitle = $('verdict-title');
    const vDesc = $('verdict-desc');
    const vGround = $('verdict-ground-truth');

    const isFake = data.decision.label.includes('FAKE');
    banner.className = 'verdict-banner ' + (isFake ? 'verdict-danger' : 'verdict-safe');

    vBadge.textContent = data.decision.label;
    vTitle.textContent = isFake ? 'Synthetic Manipulation Detected' : 'Authentic Video & Audio Verified';
    vDesc.textContent = data.decision.sub;

    const risk = data.report?.risk || (isFake ? 'high' : 'low');
    rBadge.className = 'risk-badge risk-' + risk;
    rBadge.textContent = risk.toUpperCase() + ' RISK';

    if (data.ground_truth && data.ground_truth.category) {
      const isCorrect = (isFake === data.ground_truth.any_fake);
      vGround.textContent = `Ground Truth: ${data.ground_truth.category} (${isCorrect ? '✓ Match' : '✗ Divergence'})`;
    } else {
      vGround.textContent = 'Ground Truth: Unspecified';
    }

    // 2. Animate Gauges
    setGauge('gauge-meter-video', 'gauge-val-video', 'score-video', data.scores.video);
    setGauge('gauge-meter-audio', 'gauge-val-audio', 'score-audio', data.scores.audio);
    setGauge('gauge-meter-any', 'gauge-val-any', 'score-any', data.scores.any);

    // 3. Media Player
    const player = $('result-video');
    if (player && data.video_url) {
      player.src = data.video_url;
      player.load();
    }

    // 4. Telemetry Metrics
    const info = data.info || {};
    $('info-face').textContent = info.face_prob != null ? info.face_prob.toFixed(3) : '—';
    $('info-rms').textContent = info.audio_rms != null ? info.audio_rms.toFixed(4) : '—';
    $('info-pre').textContent = (info.preprocess_s ?? '—') + ' s';
    $('info-inf').textContent = (info.inference_s ?? '—') + ' s';
    $('info-total').textContent = (data.elapsed_total ?? '—') + ' s';

    // 5. Face & Lip Crops Gallery
    renderInspectionGallery(info);

    // 6. Findings List
    lastVideoReport = data.report || null;
    const findingsList = $('report-findings');
    if (findingsList) {
      findingsList.innerHTML = '';
      (lastVideoReport?.findings || []).forEach((finding) => {
        const li = document.createElement('li');
        li.textContent = finding;
        findingsList.appendChild(li);
      });
    }

    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderInspectionGallery(info) {
    const facesStrip = $('gallery-faces-strip');
    const lipsStrip = $('gallery-lips-strip');
    const wrap = $('crops-gallery-wrap');
    if (!facesStrip || !lipsStrip) return;

    facesStrip.innerHTML = '';
    lipsStrip.innerHTML = '';

    const faces = info.face_crops || [];
    const lips = info.lip_crops || [];

    if (faces.length === 0 && lips.length === 0) {
      if (wrap) wrap.classList.add('hidden');
      return;
    }
    if (wrap) wrap.classList.remove('hidden');

    faces.forEach((cropUrl, idx) => {
      const card = document.createElement('div');
      card.className = 'crop-card';
      card.innerHTML = `
        <span class="crop-tag">#${idx + 1}</span>
        <img class="crop-img" src="${cropUrl}" alt="Face crop ${idx + 1}" loading="lazy">
      `;
      facesStrip.appendChild(card);
    });

    lips.forEach((cropUrl, idx) => {
      const card = document.createElement('div');
      card.className = 'crop-card';
      card.innerHTML = `
        <span class="crop-tag">Lip #${idx + 1}</span>
        <img class="crop-img" src="${cropUrl}" alt="Lip ROI ${idx + 1}" loading="lazy">
      `;
      lipsStrip.appendChild(card);
    });
  }

  // Gallery Tab Switching (Faces vs Lips)
  $('btn-view-faces')?.addEventListener('click', () => {
    $('btn-view-faces').classList.add('active');
    $('btn-view-lips').classList.remove('active');
    $('gallery-faces-strip').classList.remove('hidden');
    $('gallery-lips-strip').classList.add('hidden');
  });

  $('btn-view-lips')?.addEventListener('click', () => {
    $('btn-view-lips').classList.add('active');
    $('btn-view-faces').classList.remove('active');
    $('gallery-lips-strip').classList.remove('hidden');
    $('gallery-faces-strip').classList.add('hidden');
  });

  function showVideoError(msg) {
    stopPipelineStepper();
    const box = $('video-error-box');
    if (box) {
      box.classList.remove('hidden');
      $('video-error-msg').textContent = msg;
      box.scrollIntoView({ behavior: 'smooth' });
    }
  }

  // ========================================================
  // 6. Video Analysis Submissions
  // ========================================================
  async function submitVideoAnalysis(formData) {
    startPipelineStepper();
    try {
      const res = await fetch('/api/analyze', { method: 'POST', body: formData });
      const data = await res.json();
      if (!data.ok) {
        showVideoError(data.error || 'Video analysis failed.');
        return;
      }
      showVideoResults(data);
    } catch (err) {
      showVideoError('Network or server error: ' + err.message);
    }
  }

  // Upload Form
  $('upload-form')?.addEventListener('submit', (e) => {
    e.preventDefault();
    const fileInput = $('file-input');
    if (!fileInput.files.length) {
      showVideoError('Please drag & drop or select a video file first.');
      return;
    }
    const fd = new FormData();
    fd.append('mode', 'upload');
    fd.append('file', fileInput.files[0]);
    submitVideoAnalysis(fd);
  });

  // Sample Buttons
  $$('.sample-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const fd = new FormData();
      fd.append('mode', 'sample');
      fd.append('sample', btn.dataset.sample);
      submitVideoAnalysis(fd);
    });
  });

  // Export JSON Report
  $('btn-download-report')?.addEventListener('click', () => {
    if (!lastVideoReport) return;
    const blob = new Blob([JSON.stringify(lastVideoReport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `deepfake-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // ========================================================
  // 7. Interactive Audio Timeline Canvas Chart
  // ========================================================
  let hoveredWindowIndex = -1;

  function renderAudioTimeline(result) {
    const canvas = $('audio-timeline-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 220 * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = 220;
    const padL = 45;
    const padR = 25;
    const padT = 20;
    const padB = 30;

    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // Clear
    ctx.clearRect(0, 0, width, height);

    const windows = result.all_windows || [];
    const totalSec = result.analyzed_seconds || 1.0;

    // Draw Grid & Y-Axis labels (0%, 25%, 50%, 75%, 100%)
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    [0, 0.25, 0.50, 0.75, 1.0].forEach((ratio) => {
      const y = padT + plotH * (1 - ratio);
      ctx.fillText(Math.round(ratio * 100) + '%', padL - 8, y);

      ctx.beginPath();
      ctx.strokeStyle = ratio === 0.5 ? 'rgba(244, 63, 94, 0.4)' : 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = ratio === 0.5 ? 1.5 : 1;
      if (ratio === 0.5) ctx.setLineDash([4, 4]);
      else ctx.setLineDash([]);
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // 50% Threshold tag
    ctx.fillStyle = '#f43f5e';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Threshold (0.50)', padL + 6, padT + plotH * 0.5 - 6);

    if (windows.length === 0) return;

    // Draw Sliding Windows as Segment Bars
    const barWidth = Math.max(3, plotW / windows.length - 2);

    windows.forEach((win, idx) => {
      const x = padL + (win.start / totalSec) * plotW;
      const xEnd = padL + (win.end / totalSec) * plotW;
      const w = Math.max(4, xEnd - x);
      const score = Math.max(0, Math.min(1, win.fake_score));
      const barH = score * plotH;
      const y = padT + plotH - barH;

      // Color based on risk
      let color = 'rgba(16, 185, 129, 0.75)'; // Emerald
      if (score >= 0.85) {
        color = 'rgba(244, 63, 94, 0.85)'; // Crimson
      } else if (score >= 0.50) {
        color = 'rgba(245, 158, 11, 0.85)'; // Amber
      }

      if (idx === hoveredWindowIndex) {
        color = '#00e5ff'; // Highlighted
      }

      // Fill bar
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x, y, w, barH, [3, 3, 0, 0]);
      ctx.fill();

      // Peak Indicator
      if (score === result.peak_probability && score >= 0.5) {
        ctx.fillStyle = '#f43f5e';
        ctx.beginPath();
        ctx.arc(x + w / 2, y - 6, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw Playback Scrubber Line if Audio Playing
    const audioPlayer = $('result-audio-player');
    if (audioPlayer && !isNaN(audioPlayer.currentTime) && totalSec > 0) {
      const scrubX = padL + (audioPlayer.currentTime / totalSec) * plotW;
      if (scrubX >= padL && scrubX <= width - padR) {
        ctx.beginPath();
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 2;
        ctx.shadowColor = '#00e5ff';
        ctx.shadowBlur = 8;
        ctx.moveTo(scrubX, padT - 5);
        ctx.lineTo(scrubX, padT + plotH);
        ctx.stroke();
        ctx.shadowBlur = 0; // reset
      }
    }

    // Draw X-Axis Time Labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const tickSteps = Math.min(8, Math.ceil(totalSec));
    for (let i = 0; i <= tickSteps; i++) {
      const sec = (totalSec / tickSteps) * i;
      const x = padL + (sec / totalSec) * plotW;
      ctx.fillText(sec.toFixed(1) + 's', x, padT + plotH + 8);
    }
  }

  function setupTimelineInteractions() {
    const canvas = $('audio-timeline-canvas');
    const audioPlayer = $('result-audio-player');
    if (!canvas) return;

    function getWindowAtX(clientX) {
      if (!activeAudioData || !activeAudioData.all_windows) return -1;
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const padL = 45;
      const padR = 25;
      const plotW = rect.width - padL - padR;
      const totalSec = activeAudioData.analyzed_seconds || 1.0;

      const clickSec = ((x - padL) / plotW) * totalSec;
      return activeAudioData.all_windows.findIndex((win) => clickSec >= win.start && clickSec <= win.end);
    }

    canvas.addEventListener('mousemove', (e) => {
      const idx = getWindowAtX(e.clientX);
      if (idx !== hoveredWindowIndex) {
        hoveredWindowIndex = idx;
        renderAudioTimeline(activeAudioData);
      }
    });

    canvas.addEventListener('mouseleave', () => {
      hoveredWindowIndex = -1;
      if (activeAudioData) renderAudioTimeline(activeAudioData);
    });

    canvas.addEventListener('click', (e) => {
      const idx = getWindowAtX(e.clientX);
      if (idx !== -1 && activeAudioData?.all_windows[idx] && audioPlayer) {
        const targetWin = activeAudioData.all_windows[idx];
        audioPlayer.currentTime = targetWin.start;
        audioPlayer.play();
      }
    });

    // Real-time audio playback scrubber animation
    if (audioPlayer) {
      audioPlayer.addEventListener('timeupdate', () => {
        const timer = $('playback-timer');
        if (timer) {
          const cur = formatTime(audioPlayer.currentTime);
          const dur = formatTime(audioPlayer.duration || activeAudioData?.analyzed_seconds || 0);
          timer.textContent = `${cur} / ${dur}`;
        }
        if (activeAudioData) renderAudioTimeline(activeAudioData);
      });
    }

    window.addEventListener('resize', () => {
      if (activeAudioData) renderAudioTimeline(activeAudioData);
    });
  }

  function formatTime(sec) {
    if (isNaN(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  }

  // ========================================================
  // 8. Standalone Audio Results Renderer
  // ========================================================
  function showAudioResults(data) {
    $('audio-loading-card')?.classList.add('hidden');
    $('audio-error-box')?.classList.add('hidden');
    const wrap = $('audio-results-wrap');
    if (!wrap) return;
    wrap.classList.remove('hidden');

    const result = data.result;
    activeAudioData = result;

    const isFake = result.label === 'fake';
    const banner = $('audio-verdict-banner');
    const badge = $('audio-verdict-badge');
    const title = $('audio-verdict-title');
    const desc = $('audio-verdict-desc');
    const splicedBadge = $('audio-spliced-badge');

    banner.className = 'verdict-banner ' + (isFake ? 'verdict-danger' : 'verdict-safe');
    badge.textContent = isFake ? (result.is_spliced ? 'SPLICED / PARTIAL SYNTHETIC' : 'SYNTHETIC VOICE CLONE') : 'AUTHENTIC HUMAN SPEECH';
    title.textContent = isFake ? `Detected Voice Manipulation (${(result.fake_probability * 100).toFixed(1)}%)` : 'All Audio Windows Verified Authentic';
    desc.textContent = result.verdict_detail;

    splicedBadge.textContent = result.is_spliced ? 'SPLICING DETECTED' : 'CONSISTENT';
    splicedBadge.className = 'risk-badge ' + (result.is_spliced ? 'risk-high' : 'risk-low');

    $('audio-duration-tag').textContent = `${result.analyzed_seconds}s (${result.windows_analyzed} windows)`;
    $('audio-filename-meta').textContent = data.filename;

    // Load audio player
    const player = $('result-audio-player');
    if (player && data.audio_url) {
      player.src = data.audio_url;
      player.load();
    }

    // Render Timeline Chart
    renderAudioTimeline(result);

    // Populate Top Windows Diagnostics Table
    const tbody = $('top-windows-tbody');
    if (tbody) {
      tbody.innerHTML = '';
      (result.top_windows || []).forEach((win) => {
        const tr = document.createElement('tr');
        const isWinFake = win.fake_score >= 0.50;
        const tagClass = win.fake_score >= 0.85 ? 'tag-danger' : isWinFake ? 'tag-warn' : 'tag-safe';
        const tagLabel = win.fake_score >= 0.85 ? 'High Confidence Fake' : isWinFake ? 'Suspicious' : 'Authentic';

        tr.innerHTML = `
          <td><strong>${win.start_seconds}s – ${win.end_seconds}s</strong></td>
          <td><code class="metric-num">${(win.fake_score * 100).toFixed(1)}%</code></td>
          <td>${isWinFake ? 'Manipulated Speech' : 'Genuine Acoustic Profile'}</td>
          <td><span class="tag-status ${tagClass}">${tagLabel}</span></td>
          <td>
            <button type="button" class="btn btn-xs btn-secondary btn-jump" data-start="${win.start_seconds}">
              Play Segment ▶
            </button>
          </td>
        `;
        tbody.appendChild(tr);
      });

      // Jump button triggers
      tbody.querySelectorAll('.btn-jump').forEach((btn) => {
        btn.addEventListener('click', () => {
          if (player) {
            player.currentTime = parseFloat(btn.dataset.start);
            player.play();
          }
        });
      });
    }

    wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function showAudioError(msg) {
    $('audio-loading-card')?.classList.add('hidden');
    const box = $('audio-error-box');
    if (box) {
      box.classList.remove('hidden');
      $('audio-error-msg').textContent = msg;
      box.scrollIntoView({ behavior: 'smooth' });
    }
  }

  // Audio Upload Form
  $('audio-upload-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = $('audio-file-input');
    if (!input.files.length) {
      showAudioError('Please drag & drop or select an audio recording first.');
      return;
    }
    $('audio-loading-card')?.classList.remove('hidden');
    $('audio-results-wrap')?.classList.add('hidden');
    $('audio-error-box')?.classList.add('hidden');

    const fd = new FormData();
    fd.append('file', input.files[0]);

    try {
      const res = await fetch('/api/analyze-audio', { method: 'POST', body: fd });
      const data = await res.json();
      if (!data.ok) {
        showAudioError(data.error || 'Audio analysis failed.');
        return;
      }
      showAudioResults(data);
    } catch (err) {
      showAudioError('Network or server error: ' + err.message);
    }
  });

  // Audio Benchmark Sample Buttons
  $$('.btn-sample').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const sample = btn.dataset.audioSample;
      if (!sample) return;

      $('audio-loading-card')?.classList.remove('hidden');
      $('audio-results-wrap')?.classList.add('hidden');
      $('audio-error-box')?.classList.add('hidden');

      const fd = new FormData();
      fd.append('mode', 'sample');
      fd.append('sample', sample);

      try {
        const res = await fetch('/api/analyze-audio', { method: 'POST', body: fd });
        const data = await res.json();
        if (!data.ok) {
          showAudioError(data.error || 'Sample audio analysis failed.');
          return;
        }
        showAudioResults(data);
      } catch (err) {
        showAudioError('Network error: ' + err.message);
      }
    });
  });

  // ========================================================
  // 9. Initialization
  // ========================================================
  document.addEventListener('DOMContentLoaded', () => {
    initTabs();

    setupDropzone({
      dropzoneId: 'video-dropzone',
      inputId: 'file-input',
      defaultContentId: 'dropzone-content-default',
      previewWrapId: 'video-preview-wrap',
      previewElId: 'video-preview',
      clearBtnId: 'btn-clear-video',
      mediaType: 'video',
    });

    setupDropzone({
      dropzoneId: 'audio-dropzone',
      inputId: 'audio-file-input',
      defaultContentId: 'audio-dropzone-content',
      previewWrapId: 'audio-preview-wrap',
      previewElId: 'audio-preview',
      clearBtnId: 'btn-clear-audio',
      mediaType: 'audio',
    });

    setupTimelineInteractions();
  });
})();
