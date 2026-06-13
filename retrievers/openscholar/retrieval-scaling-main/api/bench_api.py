#!/usr/bin/env python3
import argparse
import statistics as stats
import time
import requests


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--n_docs", type=int, default=1)
    parser.add_argument("--domains", default="pes2o_v3")
    # Reference query from api/api_index.py:test_search
    parser.add_argument(
        "--query",
        default="when was the last time anyone was on the moon?",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/search"
    payload = {
        "query": args.query,
        "n_docs": args.n_docs,
        "domains": args.domains,
    }
    headers = {"Content-Type": "application/json"}

    # Warmup
    for _ in range(args.warmup):
        requests.post(url, json=payload, headers=headers, timeout=120)

    latencies = []
    failures = 0
    start_all = time.time()
    for _ in range(args.n):
        t0 = time.time()
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            if r.status_code != 200:
                failures += 1
        except Exception:
            failures += 1
        t1 = time.time()
        latencies.append(t1 - t0)
    total = time.time() - start_all

    avg = stats.mean(latencies)
    med = stats.median(latencies)
    p90 = percentile(latencies, 90)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    print(f"URL: {url}")
    print(f"Requests: {args.n}, Warmup: {args.warmup}, Failures: {failures}")
    print(f"Total time: {total:.2f}s, Throughput: {args.n/total:.2f} req/s")
    print(f"Latency (s): avg={avg:.3f}, p50={med:.3f}, p90={p90:.3f}, p95={p95:.3f}, p99={p99:.3f}")


if __name__ == "__main__":
    main()
