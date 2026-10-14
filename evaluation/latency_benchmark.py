import statistics
import time
from typing import Dict, List

import requests


def run_latency_benchmark(
    api_url: str = "http://127.0.0.1:8000/chat",
    warmup_requests: int = 10,
    benchmark_requests: int = 100,
) -> Dict[str, float]:
    payload = {
        "session_id": "latency-benchmark-session",
        "message": "aaj dinner mein kya khau for low calorie?",
        "nutrition_vector": [0.2, 0.5, 0.1, 0.0, 0.3, 0.4, 0.2, 0.1],
        "calorie_budget": 450.0,
    }

    for _ in range(warmup_requests):
        requests.post(api_url, json=payload, timeout=30)

    latencies: List[float] = []
    for _ in range(benchmark_requests):
        start = time.perf_counter()
        resp = requests.post(api_url, json=payload, timeout=30)
        resp.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000.0)

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(0.50 * len(latencies_sorted))]
    p95 = latencies_sorted[max(0, int(0.95 * len(latencies_sorted)) - 1)]
    p99 = latencies_sorted[max(0, int(0.99 * len(latencies_sorted)) - 1)]

    results = {
        "count": float(len(latencies)),
        "mean_ms": float(statistics.mean(latencies)),
        "stdev_ms ": float(statistics.pstdev(latencies)),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(max(latencies)),
    }
    return results


if __name__ == "__main__":
    metrics = run_latency_benchmark()
    print("Latency Benchmark Results")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

