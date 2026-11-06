#!/usr/bin/env bash
# run_e8_bench.sh — DiskBBQ residual codes compared in KnnIndexTester: OSQ, ASH and the experimental E8 codebook.
#
# Every run uses the same IVF index structure (clustering, SOAR spilling, probing and rescoring); only the residual
# code changes. Operating points (rate R = bits per input dimension):
#
#   name        quantization_type  quantize_bits  projected_dims_fraction   setting            R
#   osq1        osq                1              -                         full-dimensional   1   (DiskBBQ default)
#   ash1_full   ash                1              1.0                       full-dimensional   1   (sign bits after rotation)
#   e8r_full    e8                 1              1.0                       full-dimensional   1   (E8 roots)
#   ash2        ash                2              0.5                       projected          1   (ASH default)
#   e8t         e8                 2              0.5                       projected          1   (E8 two-stage)
#   ash1        ash                1              0.5                       projected          1/2
#   e8r         e8                 1              0.5                       projected          1/2
#   ash2_float  ash                2 (float query) 0.5                      projected          1   (speed reference, see below)
#
# Speed: the E8 scorer is a scalar Java reference (float table lookup); ASH and OSQ use tuned integer kernels. So
# compare RECALL and VISITED across codes; QPS is only comparable between e8* and ash2_float (both float-query paths).
#
# Usage (from the worktree root, i.e. the directory that contains this repo's gradlew):
#   qa/vector/run_e8_bench.sh                          # all datasets, all codes
#   DATASETS="fiqa-gte-base" CODES="osq1 e8r_full" qa/vector/run_e8_bench.sh
#   REINDEX=false qa/vector/run_e8_bench.sh            # reuse indices built by a previous run
#
# Output: qa/vector/target/knn_results/results.csv (one row per search configuration, appended by KnnIndexTester),
#         qa/vector/target/e8_bench_logs/<dataset>_<code>.log, and a summary from summarize_e8_bench.py.

set -euo pipefail

WT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VEC_DIR="$WT_DIR/qa/vector"
# The datasets live in the main checkout's qa/vector/.data; this worktree sits inside it at .data/es-e8.
DATA_DIR="${DATA_DIR:-$(cd "$WT_DIR/.." && pwd)}"
LOG_DIR="$VEC_DIR/target/e8_bench_logs"
TMP_DIR="$VEC_DIR/target/e8_bench_tmp"
mkdir -p "$LOG_DIR" "$TMP_DIR"

REINDEX="${REINDEX:-true}"
# Indices hold the float vectors too (about 4 bytes x dims x docs, so ~3 GB per code for 1M 768-d vectors). By default
# each index is deleted after its run, so only one index is on disk at a time; KEEP_INDEX=true keeps them (needed for
# REINDEX=false on a later run).
KEEP_INDEX="${KEEP_INDEX:-false}"
# Each code builds its own index, and index building is not deterministic (k-means, threads, merges): at low visit
# percentages recall moved by up to ~4.6 points between two builds of the same configuration. REPEATS rebuilds and
# searches every (dataset, code) that many times; summarize_e8_bench.py averages them and reports the spread.
REPEATS="${REPEATS:-1}"
# Default suite: 12 datasets, 10 text-embedding (4 models) and 2 non-text with Euclidean distance. trec-covid-jinav3 is
# left out (50 queries). hotpotqa-* (5.2M vectors) and dbpedia-entity-gte-base (4.6M) dominate the run time; set
# MAX_DOCS (e.g. 1000000) to index only the first vectors of every dataset.
DATASETS="${DATASETS:-fiqa-E5-small fiqa-arctic fiqa-gte-base quora-E5-small quora-arctic quora-gte-base hotpotqa-E5-small hotpotqa-gte-base dbpedia-entity-gte-base wiki-cohere ann-gist-1M fashion-minst}"
MAX_DOCS="${MAX_DOCS:-0}"
CODES="${CODES:-osq1 ash1_full e8r_full ash2 e8t ash1 e8r}"
VISITS="${VISITS:-0.5, 1.0, 2.0, 4.0, 100.0}"
OVERSAMPLE="${OVERSAMPLE:-1.0, 3.0}"
THREADS="${THREADS:-8}"
HEAP="${HEAP:-16g}"
MAX_QUERIES="${MAX_QUERIES:-1000}"

code_params() {   # prints: quantization_type quantize_bits query_quantize_bits projected_dims_fraction
    case "$1" in
        osq1) echo "osq 1 null 0.5" ;;
        ash1_full) echo "ash 1 4 1.0" ;;
        e8r_full) echo "e8 1 4 1.0" ;;
        ash2) echo "ash 2 4 0.5" ;;
        e8t) echo "e8 2 4 0.5" ;;
        ash1) echo "ash 1 4 0.5" ;;
        e8r) echo "e8 1 4 0.5" ;;
        ash2_float) echo "ash 2 0 0.5" ;;
        *) echo "unknown code: $1" >&2; exit 1 ;;
    esac
}

for ds in $DATASETS; do
    dsdir="$DATA_DIR/$ds"
    if [[ "$ds" == "fashion-minst" ]]; then
        corpus_files="$dsdir/fashion-minst-train.fvec"; queries="$dsdir/fashion-minst-test.fvec"
    else
        corpus_files=$(ls "$dsdir"/corpus-*.fvec | sort); queries=$(ls "$dsdir"/queries-*.fvec | sort | head -1)
    fi
    case "$ds" in ann-gist-1M|fashion-minst) space="euclidean" ;; *) space="cosine" ;; esac
    docs=$(echo "$corpus_files" | python3 -c 'import sys, json; print(", ".join(json.dumps(l.strip()) for l in sys.stdin if l.strip()))')
    # KnnIndexTester defaults to 1000 docs for file datasets, so pass the real counts (fvec: int dim + dim floats per row)
    count_vecs() { python3 -c 'import os, struct, sys; t = 0
for p in sys.argv[1:]:
    d = struct.unpack("<i", open(p, "rb").read(4))[0]; t += os.path.getsize(p) // (4 * (d + 1))
print(t)' "$@"; }
    num_docs=$(count_vecs $corpus_files)
    if (( MAX_DOCS > 0 && num_docs > MAX_DOCS )); then num_docs=$MAX_DOCS; fi
    num_queries=$(count_vecs "$queries")
    if (( num_queries > MAX_QUERIES )); then num_queries=$MAX_QUERIES; fi
    for code in $CODES; do
        read -r qtype bits qbits fraction <<< "$(code_params "$code")"
        cfg="$TMP_DIR/${ds}_${code}.json"
        qbits_line=""
        if [[ "$qbits" != "null" ]]; then qbits_line="\"query_quantize_bits\": $qbits,"; fi
        cat > "$cfg" <<JSON
{
  "dataset": { "file": { "doc_vectors": [$docs], "query_vectors": "$queries" } },
  "index_type": "ivf",
  "quantization_type": "$qtype",
  "quantize_bits": $bits,
  $qbits_line
  "projected_dims_fraction": $fraction,
  "vector_space": "$space",
  "dimensions": -1,
  "index_threads": $THREADS,
  "search_threads": $THREADS,
  "num_docs": $num_docs,
  "num_queries": $num_queries,
  "reindex": $REINDEX,
  "force_merge": true,
  "force_merge_max_num_segments": 1,
  "visit_percentage": [$VISITS],
  "over_sampling_factor": [$OVERSAMPLE],
  "k": [10]
}
JSON
        for rep in $(seq "$REPEATS"); do
            log="$LOG_DIR/${ds}_${code}_r${rep}.log"
            echo "== $ds $code rep $rep ($qtype, bits=$bits, query bits=$qbits, fraction=$fraction) -> $log"
            (cd "$WT_DIR" && ./gradlew :qa:vector:checkVec --no-daemon --console=plain \
                -Dbench.heap="$HEAP" "--args=$cfg") > "$log" 2>&1 \
                || echo "   FAILED (see $log)"
            if grep -q "No space left on device" "$log"; then echo "   disk full: free space or lower MAX_DOCS" >&2; fi
            if [[ "$KEEP_INDEX" != "true" ]]; then rm -rf "$VEC_DIR"/target/knn_index/*; fi
        done
    done
done

python3 "$VEC_DIR/summarize_e8_bench.py" "$VEC_DIR/target/knn_results/results.csv" | tee "$VEC_DIR/target/e8_bench_summary.txt"
