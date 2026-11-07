"""Benchmark inference latency, throughput, model loading time, and memory for audio detectors.

Measures:
- Model instantiation and weight loading time (ms)
- Inference latency per audio duration (3s, 10s, 30s)
- Real-Time Factor (RTF = compute time / audio duration)
- Throughput (audio seconds processed per second)
- Peak memory usage (RAM / VRAM)

Usage:
    python scripts/benchmark_audio.py [--models aasist legacy_wav2vec_mlp] [--device cpu] [--output reports/audio_benchmarks.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepfake_detector.audio_detection.registry import get_detector, list_detectors
from deepfake_detector.audio_detection.service import StandaloneAudioService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_audio")


def measure_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024**2)
    except Exception:
        return 0.0


def benchmark_model(
    model_id: str,
    device: str = "cpu",
    durations: List[float] = [3.0, 10.0, 30.0],
    repeats: int = 5,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    logger.info("=== Benchmarking '%s' on %s ===", model_id, device)

    # 1. Measure load time & initial memory
    mem_before = measure_memory_mb()
    t_start_load = time.perf_counter()
    try:
        detector = get_detector(model_id, device=device)
        load_time_ms = (time.perf_counter() - t_start_load) * 1000.0
        mem_after_load = measure_memory_mb()
    except Exception as e:
        logger.error("Failed to load model '%s': %s", model_id, e)
        return {
            "model_id": model_id,
            "status": "failed",
            "error": str(e),
        }

    service = StandaloneAudioService(detector, batch_size=8, vad_enabled=False)

    # 2. Warmup
    warmup_wave = np.random.randn(sample_rate * 3).astype(np.float32) * 0.1
    windows = service.window_generator.generate_windows(warmup_wave)
    if windows:
        waves = np.stack([w.data for w in windows])
        lengths = [w.valid_length_samples for w in windows]
        _ = detector.predict_batch(waves, lengths=lengths)

    duration_results = {}

    for dur in durations:
        n_samples = int(dur * sample_rate)
        latencies = []

        for _ in range(repeats):
            # Synthetic speech-like signal
            t_wave = np.linspace(0, dur, n_samples)
            wave = (0.3 * np.sin(2 * np.pi * 220 * t_wave) + 0.1 * np.random.randn(n_samples)).astype(np.float32)

            wins = service.window_generator.generate_windows(wave)
            if not wins:
                continue

            waves = np.stack([w.data for w in wins])
            lengths = [w.valid_length_samples for w in wins]

            t0 = time.perf_counter()
            _ = detector.predict_batch(waves, lengths=lengths)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        if latencies:
            mean_lat = float(np.mean(latencies))
            std_lat = float(np.std(latencies))
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))

            # Real Time Factor: compute_seconds / audio_seconds
            rtf = (mean_lat / 1000.0) / dur
            throughput = dur / (mean_lat / 1000.0) if mean_lat > 0 else 0.0

            duration_results[f"{dur}s"] = {
                "audio_duration_seconds": dur,
                "latency_mean_ms": round(mean_lat, 2),
                "latency_std_ms": round(std_lat, 2),
                "latency_p50_ms": round(p50, 2),
                "latency_p95_ms": round(p95, 2),
                "rtf": round(rtf, 4),
                "throughput_audio_sec_per_sec": round(throughput, 2),
            }
            logger.info(
                "  Duration %4.1fs | Latency: %6.1f ± %4.1f ms | RTF: %6.4f | Throughput: %5.1fx",
                dur, mean_lat, std_lat, rtf, throughput
            )

    mem_end = measure_memory_mb()

    return {
        "model_id": model_id,
        "status": "success",
        "device": device,
        "load_time_ms": round(load_time_ms, 2),
        "memory_load_delta_mb": round(mem_after_load - mem_before, 2),
        "memory_total_rss_mb": round(mem_end, 2),
        "durations": duration_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark audio detector runtime performance.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["legacy_wav2vec_mlp", "aasist"],
        help="List of model IDs to benchmark.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run benchmarks on ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=[3.0, 10.0, 30.0],
        help="Audio durations in seconds to test.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of inference repetitions for averaging.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "benchmark_audio.json",
        help="Path to save output JSON benchmark results.",
    )
    args = parser.parse_args()

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "benchmarks": {},
    }

    for m in args.models:
        res = benchmark_model(
            model_id=m,
            device=args.device,
            durations=args.durations,
            repeats=args.repeats,
        )
        results["benchmarks"][m] = res

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved benchmark results to: %s", args.output)


if __name__ == "__main__":
    main()
