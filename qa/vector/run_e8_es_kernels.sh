#!/usr/bin/env bash
# run_e8_es_kernels.sh — the E8 residual code inside Elasticsearch's IVF formats, with the Panama and the native scan
# kernels, against the codes it would replace, through KnnIndexTester (index, force-merge to one segment, search).
#
# Codes (all at 1 bit per input dimension unless noted):
#
#   name        format                quantization_type  bits  precondition  setting
#   osq1        ESNext DiskBBQ        osq                1     false         ES default (D1Q4)
#   osq1p       ESNext DiskBBQ        osq                1     true          OSQ with the same rotation E8 uses
#   osq_e8      ESNext DiskBBQ        osq_e8             1     true          E8 roots in place of OSQ (QuantEncoding.E8_ROOTS)
#   osq1p_fold  ESNext DiskBBQ        osq                1     true          osq1p + leaf-centroid fold (writer-only, see
#   osq_e8_fold ESNext DiskBBQ        osq_e8             1     true          osq_e8 + fold     E8ResidualQuantizer#FOLD_LEAF_CENTROID)
#   ash2        ESNext ASH            ash                2     -             ASH default: 2 bits on D/2 projected dims
#   e8t         ESNext ASH            e8                 2     -             E8 two-stage on D/2
#   ash1_full   ESNext ASH            ash                1     -             sign bits on all D dims
#   e8r_full    ESNext ASH            e8                 1     -             E8 roots on all D dims
#
# Kernels (KERNELS): "native" uses the native library for every code (OSQ: vec_dotd1q4_bulk..., E8: vec_e8_lookup_bulk);
# "panama" disables the native library (-Dorg.elasticsearch.nativeaccess.enableVectorLibrary=false) so every code
# runs its Java Vector API kernel. Both are run with the same index (REINDEX=false for the second kernel).
#
# Before the first run, build the native library from this worktree's sources (it contains the new E8 kernel):
#   VEC_NATIVE_BUILD=host ./gradlew :libs:simdvec:...   (see README_E8_ES.md, "Build")
#
# Usage, from the worktree root:
#   qa/vector/run_e8_es_kernels.sh
#   DATASETS="fiqa-gte-base" CODES="osq1p osq_e8" KERNELS="native panama" qa/vector/run_e8_es_kernels.sh
#
# Output: qa/vector/target/knn_results/results.csv (appended by KnnIndexTester; the config file name carries the
# dataset, code and kernel), logs in qa/vector/target/e8_kernel_logs, and a summary from summarize_e8_es_kernels.py.

set -euo pipefail

WT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VEC_DIR="$WT_DIR/qa/vector"
DATA_DIR="${DATA_DIR:-$(cd "$WT_DIR/.." && pwd)}"
LOG_DIR="$VEC_DIR/target/e8_kernel_logs"
TMP_DIR="$VEC_DIR/target/e8_kernel_tmp"
mkdir -p "$LOG_DIR" "$TMP_DIR"

DATASETS="${DATASETS:-fiqa-gte-base quora-E5-small wiki-cohere hotpotqa-gte-base}"
CODES="${CODES:-osq1 osq1p osq_e8 osq1p_fold osq_e8_fold ash2 e8t ash1_full e8r_full}"
KERNELS="${KERNELS:-native panama}"
MAX_DOCS="${MAX_DOCS:-500000}"
MAX_QUERIES="${MAX_QUERIES:-1000}"
VISITS="${VISITS:-0.5, 1.0, 2.0, 4.0, 100.0}"
OVERSAMPLE="${OVERSAMPLE:-1.0, 3.0}"
INDEX_THREADS="${INDEX_THREADS:-8}"
# one search thread: avg_latency_ms is then the single-query latency, comparable across codes and kernels
SEARCH_THREADS="${SEARCH_THREADS:-1}"
HEAP="${HEAP:-16g}"
REPEATS="${REPEATS:-1}"

code_params() {   # quantization_type quantize_bits projected_dims_fraction precondition fold_leaf_centroid
    case "$1" in
        osq1) echo "osq 1 0.5 false false" ;;
        osq1p) echo "osq 1 0.5 true false" ;;
        osq_e8) echo "osq_e8 1 0.5 true false" ;;
        osq1p_fold) echo "osq 1 0.5 true true" ;;
        osq_e8_fold) echo "osq_e8 1 0.5 true true" ;;
        ash2) echo "ash 2 0.5 false false" ;;
        e8t) echo "e8 2 0.5 false false" ;;
        ash1_full) echo "ash 1 1.0 false false" ;;
        e8r_full) echo "e8 1 1.0 false false" ;;
        *) echo "unknown code: $1" >&2; exit 1 ;;
    esac
}

count_vecs() { python3 -c 'import os, struct, sys; t = 0
for p in sys.argv[1:]:
    d = struct.unpack("<i", open(p, "rb").read(4))[0]; t += os.path.getsize(p) // (4 * (d + 1))
print(t)' "$@"; }

for ds in $DATASETS; do
    dsdir="$DATA_DIR/$ds"
    corpus_files=$(ls "$dsdir"/corpus-*.fvec | sort); queries=$(ls "$dsdir"/queries-*.fvec | sort | head -1)
    docs=$(echo "$corpus_files" | python3 -c 'import sys, json; print(", ".join(json.dumps(l.strip()) for l in sys.stdin if l.strip()))')
    num_docs=$(count_vecs $corpus_files); if (( MAX_DOCS > 0 && num_docs > MAX_DOCS )); then num_docs=$MAX_DOCS; fi
    num_queries=$(count_vecs "$queries"); if (( num_queries > MAX_QUERIES )); then num_queries=$MAX_QUERIES; fi
    for code in $CODES; do
        read -r qtype bits fraction precond fold <<< "$(code_params "$code")"
        for rep in $(seq "$REPEATS"); do
            first=true
            for kernel in $KERNELS; do
                cfg="$TMP_DIR/${ds}__${code}__${kernel}__r${rep}.json"
                # the first kernel builds the index, the next ones search the same index
                reindex=$([[ "$first" == "true" ]] && echo true || echo false)
                cat > "$cfg" <<JSON
{
  "dataset": { "file": { "doc_vectors": [$docs], "query_vectors": "$queries" } },
  "index_type": "ivf",
  "quantization_type": "$qtype",
  "quantize_bits": $bits,
  "projected_dims_fraction": $fraction,
  "precondition": $precond,
  "preconditioning_block_dims": 32,
  "vector_space": "cosine",
  "dimensions": -1,
  "index_threads": $INDEX_THREADS,
  "search_threads": $SEARCH_THREADS,
  "num_docs": $num_docs,
  "num_queries": $num_queries,
  "reindex": $reindex,
  "force_merge": true,
  "force_merge_max_num_segments": 1,
  "visit_percentage": [$VISITS],
  "over_sampling_factor": [$OVERSAMPLE],
  "k": [10]
}
JSON
                # the fold only changes what the writer stores, so it only matters for the run that builds the index
                props=("-Des.vectors.diskbbq.fold_leaf_centroid=$fold")
                if [[ "$kernel" == "panama" ]]; then
                    props+=("-Dorg.elasticsearch.nativeaccess.enableVectorLibrary=false" "-Des.vectors.e8.kernel=panama")
                else
                    props+=("-Des.vectors.e8.kernel=native")
                fi
                log="$LOG_DIR/${ds}__${code}__${kernel}__r${rep}.log"
                echo "== $ds $code ($qtype bits=$bits fraction=$fraction precondition=$precond) kernel=$kernel rep=$rep -> $log"
                (cd "$WT_DIR" && ./gradlew :qa:vector:checkVec --no-daemon --console=plain \
                    -Dbench.heap="$HEAP" "${props[@]}" "--args=$cfg") > "$log" 2>&1 \
                    || echo "   FAILED (see $log)"
                grep -E "E8|Using native vector library|Java vector incubator API enabled" "$log" | head -3 | sed 's/^/   /' || true
                first=false
            done
            rm -rf "$VEC_DIR"/target/knn_index/*
        done
    done
done

python3 "$VEC_DIR/summarize_e8_es_kernels.py" "$VEC_DIR/target/knn_results/results.csv" | tee "$VEC_DIR/target/e8_kernel_summary.txt"
