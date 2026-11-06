#!/usr/bin/env python3
"""Summarizes run_e8_bench.sh results from KnnIndexTester's results.csv.

Prints, per dataset, one table per oversampling factor: recall@10 and vectors visited per query for every residual
code at each visit percentage, plus index and force-merge time. The code is recovered from the index name, which
KnnIndexTester derives from the configuration (ivf = OSQ, ash-..., e8-..., with -p1.00 for full-dimensional and
-q0 for a float query).

  python3 summarize_e8_bench.py qa/vector/target/knn_results/results.csv
"""
import csv
import re
import sys
from collections import OrderedDict

ORDER = ["osq1", "ash1_full", "e8r_full", "ash2", "e8t", "ash2_float", "ash1", "e8r"]
LABEL = {
    "osq1": "OSQ 1-bit (full-dim, R=1)",
    "ash1_full": "ASH 1-bit (full-dim, R=1)",
    "e8r_full": "E8 roots (full-dim, R=1)",
    "ash2": "ASH 2-bit (proj, R=1)",
    "e8t": "E8 two-stage (proj, R=1)",
    "ash2_float": "ASH 2-bit float q (proj)",
    "ash1": "ASH 1-bit (proj, R=1/2)",
    "e8r": "E8 roots (proj, R=1/2)",
}


def code_of(row):
    name = row["index_name"]
    m = re.search(r"-(ivf|ash|e8)-(\d+)-(\d+)", name)
    if not m:
        return None
    kind = m.group(1)
    if kind == "ivf":
        return "osq%s" % row["quantize_bits"]
    bits = m.group(3)
    full = "-p1.00" in name
    floatq = "-q0" in name
    if kind == "ash":
        if floatq:
            return "ash%s_float" % bits
        return "ash%s%s" % (bits, "_full" if full else "")
    return ("e8r" if bits == "1" else "e8t") + ("_full" if full else "")


def dataset_of(row):
    m = re.search(r"(?:corpus-)?([^/]+?)(?:-\d+)?\.fvec", row["index_name"])
    return m.group(1) if m else row["index_name"]


def main(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    # all rows for each (dataset, code, visit, oversample): repeated builds (REPEATS, reruns) are averaged
    latest = OrderedDict()
    for r in rows:
        code = code_of(r)
        if code is None:
            continue
        key = (dataset_of(r), code, float(r["visit_percentage"] or 0), float(r["over_sampling_factor"] or 0))
        latest.setdefault(key, []).append(r)
    datasets = sorted({k[0] for k in latest})
    for ds in datasets:
        keys = [k for k in latest if k[0] == ds]
        codes = [c for c in ORDER if any(k[1] == c for k in keys)]
        visits = sorted({k[2] for k in keys})
        oversamples = sorted({k[3] for k in keys})
        for os_ in oversamples:
            print("\n== %s | oversample %g | recall@10 (vectors visited per query)" % (ds, os_))
            print("  %-28s" % "code" + "".join("%20s" % ("visit %g%%" % v) for v in visits) + "%12s%12s" % ("index s", "merge s"))
            for c in codes:
                line = "  %-28s" % LABEL.get(c, c)
                any_row = None
                for v in visits:
                    rs = latest.get((ds, c, v, os_))
                    if not rs:
                        line += "%20s" % "-"
                        continue
                    any_row = r = rs[-1]
                    rec = [float(x["recall"]) for x in rs]
                    mean = sum(rec) / len(rec)
                    vis = sum(float(x["visited"] or 0) for x in rs) / len(rs)
                    cell = "%.4f (%.0f)" % (mean, vis) if len(rs) == 1 else "%.4f±%.3f n%d" % (mean, (max(rec) - min(rec)) / 2, len(rs))
                    line += "%20s" % cell
                if any_row is not None:
                    line += "%12.0f%12.0f" % (
                        float(any_row["index_time_ms"] or 0) / 1000,
                        float(any_row["force_merge_time_ms"] or 0) / 1000,
                    )
                print(line)
    print(
        "\nCells: mean recall@10 (vectors visited), or mean±half-range over n index builds. "
        "Recall and visited are comparable across codes. QPS is not: E8 uses a scalar float-table scorer, "
        "comparable only with ash2_float."
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qa/vector/target/knn_results/results.csv")
