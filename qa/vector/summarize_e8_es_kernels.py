#!/usr/bin/env python3
"""Summarizes run_e8_es_kernels.sh results from KnnIndexTester's results.csv.

Rows are identified by the config file name, <dataset>__<code>__<kernel>__r<rep>.json. Prints, per dataset:
  1. recall@10 and single-thread latency for every (code, kernel) at each visit percentage and oversampling factor;
  2. paired comparisons, E8 code vs the code it replaces, per kernel: recall difference and latency ratio;
  3. the lowest latency at which each (code, kernel) reaches a recall target, over the (visit, oversample) grid.

  python3 summarize_e8_es_kernels.py qa/vector/target/knn_results/results.csv
"""
import csv
import os
import re
import sys
from collections import defaultdict
from statistics import mean

PAIRS = [
    ("osq_e8", "osq1p"),
    ("osq_e8", "osq1"),
    ("osq1p_fold", "osq1p"),
    ("osq_e8_fold", "osq1p_fold"),
    ("osq_e8_fold", "osq1"),
    ("e8t", "ash2"),
    ("e8r_full", "ash1_full"),
]
ORDER = ["osq1", "osq1p", "osq1p_fold", "osq_e8", "osq_e8_fold", "ash1_full", "e8r_full", "ash2", "e8t"]
TARGETS = [0.80, 0.90, 0.95]


def parse(path):
    data = defaultdict(list)  # (ds, code, kernel, visit, os) -> [(recall, latency_ms, qps, index_ms, merge_ms)]
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        m = re.match(r"(.+?)__(.+?)__(.+?)__r(\d+)\.json$", os.path.basename(r["config_file"] or ""))
        if not m:
            continue
        ds, code, kernel, _ = m.groups()
        key = (ds, code, kernel, float(r["visit_percentage"] or 0), float(r["over_sampling_factor"] or 0))
        f = lambda k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
        data[key].append((f("recall"), f("avg_latency_ms"), f("qps"), f("index_time_ms"), f("force_merge_time_ms")))
    return data


def agg(rows, i):
    return mean(x[i] for x in rows)


def main(path):
    data = parse(path)
    for ds in sorted({k[0] for k in data}):
        keys = [k for k in data if k[0] == ds]
        codes = [c for c in ORDER if any(k[1] == c for k in keys)] + sorted({k[1] for k in keys} - set(ORDER))
        kernels = sorted({k[2] for k in keys})
        visits = sorted({k[3] for k in keys})
        overs = sorted({k[4] for k in keys})
        print("\n" + "=" * 100 + "\n%s" % ds)
        for os_ in overs:
            print("\n-- oversample %g: recall@10 / avg latency ms (1 search thread)" % os_)
            print("  %-12s %-7s" % ("code", "kernel") + "".join("%18s" % ("visit %g%%" % v) for v in visits))
            for c in codes:
                for kn in kernels:
                    cells = []
                    for v in visits:
                        rows = data.get((ds, c, kn, v, os_))
                        cells.append("%18s" % ("%.4f / %.3f" % (agg(rows, 0), agg(rows, 1)) if rows else "-"))
                    if any(x.strip() != "-" for x in cells):
                        print("  %-12s %-7s" % (c, kn) + "".join(cells))
        print("\n-- paired: E8 code minus baseline, recall points; latency ratio E8 / baseline (same kernel)")
        for a, b in PAIRS:
            for kn in kernels:
                parts = []
                for os_ in overs:
                    for v in visits:
                        ra, rb = data.get((ds, a, kn, v, os_)), data.get((ds, b, kn, v, os_))
                        if ra and rb:
                            parts.append("v%g/os%g: %+.1f pt, x%.2f" % (v, os_, 100 * (agg(ra, 0) - agg(rb, 0)), agg(ra, 1) / agg(rb, 1)))
                if parts:
                    print("  %-9s vs %-9s %-7s " % (a, b, kn) + " | ".join(parts))
        print("\n-- lowest latency (ms) reaching recall@10 >= target, over visit x oversample")
        print("  %-12s %-7s" % ("code", "kernel") + "".join("%12s" % ("R>=%.2f" % t) for t in TARGETS))
        for c in codes:
            for kn in kernels:
                pts = [(agg(rows, 0), agg(rows, 1)) for (d, cc, kk, v, o), rows in data.items() if d == ds and cc == c and kk == kn]
                if not pts:
                    continue
                cells = []
                for t in TARGETS:
                    ok = [lat for rec, lat in pts if rec >= t]
                    cells.append("%12s" % ("%.3f" % min(ok) if ok else "-"))
                print("  %-12s %-7s" % (c, kn) + "".join(cells))
        print("\n-- build: index / force-merge seconds (first kernel's run builds the index)")
        for c in codes:
            rows = [x for (d, cc, kk, v, o), rs in data.items() if d == ds and cc == c for x in rs if x[3] > 0]
            if rows:
                print("  %-12s index %.1f s, merge %.1f s" % (c, agg(rows, 3) / 1000, agg(rows, 4) / 1000))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qa/vector/target/knn_results/results.csv")
