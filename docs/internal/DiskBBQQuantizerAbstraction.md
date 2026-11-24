# DiskBBQ: quantizer-agnostic IVF (design draft)

Status: draft for discussion. Scope: the `ESNext*` formats only (legacy ES920/940/950 are untouched).
On-disk layout of the `ESNext*` formats may change freely (`VERSION_CURRENT = 1`, no BWC guarantee).

## 1. Goal

Today OSQ, ASH (and the E8 PoC) each need their own scorer, bulk writer, reader/visitor code and calibration
hooks. ASH is a parallel stack; E8 is a branch inside the OSQ stack. The goal is one IVF format/reader/writer
skeleton that is parametrized by a **quantization scheme**, so that adding a quantizer means implementing a scheme,
not forking the stack.

## 2. The model (from "Configuring Project-then-Quantize Embedding Compression")

Every scheme is the same encoder: landmark `mu` (IVF centroid), residual `x - mu`, a stored gain `||x - mu||`,
a projection `U` (D -> d), a rotation `R`, and a direction quantizer `quantS` over a codebook with `b` bits.
Scoring is `gain * <Wq, v> + <q, mu>`. Distortion follows `2(1 - rho) + rho * c_b(d)`: the subspace (`rho`) and the
codebook (`c_b(d)`) decouple.

| Scheme | U (D -> d) | R | codebook | gain |
|---|---|---|---|---|
| OSQ | identity | optional block rotation (`Preconditioner`) | per-vector optimized scalar grid | interval (lower, upper) |
| ASH | learned PCA | learned (Procrustes) | uniform mid-riser grid | `scale` |
| E8 (PoC) | identity | `Preconditioner` (required) | 240 E8 roots, 8-dim blocks | `alpha` |

## 3. Canonical scoring form (verified by `CanonicalScoreFormTests`)

```
score_i = s_i * (alpha * r_i + beta * sigma_i + gamma) + l_i * kappa + o_i + omega
```

* `r_i`: raw dot product of doc code and query code (computed by the codebook's SIMD/native kernel).
* Per-doc lanes `(s, sigma, l, o)`: written by the doc encoder. `sigma` (code sum) and `l` are optional per scheme.
  `o` is similarity specific (written at index time; Euclidean folds `||x-c||^2`, `<x,c>` into it).
* Per-query scalars `(alpha, beta, gamma, kappa, omega)`: produced per posting list. Euclidean's factor `-2` is
  folded into the query scalars so doc lanes stay query independent.
* OSQ: lanes `(s, sigma, l, o)`, `gamma = 0`. ASH: lanes `(s, sigma, o)`, `kappa` unused, `gamma = -constantCorrection`.
  ASH's Euclidean `vecCentroidDot` / `vecCentroidSqDist` lanes fold into `o` (5 lanes / 20 B -> 3 lanes / 12 B).

## 4. Proposed seams

```java
/** One quantization scheme. Registered by a persisted id; selected by the mapper's quantization_type. */
interface QuantizerScheme {
    int id();
    QuantizerConfig parseConfig(...);                 // replaces sealed IvfSegmentConfig.QuantConfig
    FieldQuantizer forField(FieldContext ctx);        // dims, similarity, bits, preconditioning, config
    CalibrationQuantizer calibration();               // already exists in calib-quantizer-abstraction worktree
    CostProfile costProfile(FieldContext ctx);        // bytes/vector, lanes: feeds the cost model
}

interface FieldQuantizer {
    LatentProjection projection();                    // identity for OSQ/E8; learned W for ASH; may be null
    CorrectionSchema schema();                        // which of s / sigma / l / o lanes exist
    DocEncoder newDocEncoder();                       // residual -> code bytes + DocCorrections(s, sigma, l, o)
    QueryEncoder newQueryEncoder();                   // see below
    ScoreKernel newKernel(IndexInput in, int bulkSize);
    void writeMeta(IndexOutput out);                  // scheme-specific blob, length prefixed in .mivf
}

interface QueryEncoder {
    PreparedQuery prepare(float[] query);             // once per search (ASH: project + quantize once)
}
interface PreparedQuery {
    QueryScalars forPostingList(CentroidView c);      // per list (OSQ: quantize q - c, 16-entry cache; ASH: q.c)
    ScoringQuery codes();
}

interface ScoreKernel {
    float scoreBulk(PreparedQuery q, QueryScalars s, int count, float[] scores);       // fused codes + lanes
    float scoreBulkMasked(PreparedQuery q, QueryScalars s, int mask, float[] scores, int count);
    int minDocsToScoreInBulk();                       // E8: 1 (no individual path); OSQ/ASH: 2
}
```

Design decisions embedded above:

1. **Two projection scopes, not one.** The existing `Preconditioner` is a *space rotation*: it is applied to the stored
   raw vectors and to queries outside the codec (`IVFKnn*Query`), preserves dot/L2, and benefits rescoring and every
   scheme. ASH's `W` (D -> d) is a *latent projection* applied only to residuals inside the quantizer and does not
   preserve similarity. They stay separate; `LatentProjection` lives in `FieldQuantizer` and is serialized in the
   existing preconditioner slot of `.cenivf`. The merge policy ("inherit the learned W with most vectors, warm start")
   becomes `LatentProjection.inheritForMerge(...)`.
2. **Query encoding has two strategies**, hidden behind `PreparedQuery.forPostingList`: per-centroid residual
   quantization (OSQ, E8) and global latent quantization plus exact `<q, mu>` (ASH, the paper's form).
3. **The kernel is the performance boundary.** Existing native/Panama code-dot kernels are kept as they are. Only the
   correction fold is generalized, specialized per lane set (`{s,sigma,l,o}` is exactly today's OSQ fold, so OSQ pays
   nothing), and ASH moves from a per-vector Java loop to the fused fold.
4. **Meta stays byte-for-byte as it is today.** Each scheme keeps its current `.mivf` layout (OSQ: `QuantEncoding` id,
   rescore oversample, byte-centroid flag; ASH: bits per dimension). The scheme owns reading and writing its own part
   through the shared field-entry base, so no persisted scheme id is added. A segment's scheme is still identified by its
   codec name, as today. See "Format stability" below.
5. **Format classes**: one generic `ESNextDiskBBQVectorsFormat/Reader/Writer` parametrized by scheme. The ASH format
   class (name registered in SPI) can remain as a thin factory over the same classes at first, so segment codec names do
   not change; merging the names is a later, separate step.

Calibration (`CalibrationQuantizer`, from the `calib-quantizer-abstraction` worktree) is exposed by the scheme as is:
its `Candidate`, `storageCost`, `queryCost`, `measureError` and `quantConfig` map cleanly onto `QuantizerScheme`.
Later, the distortion law (`rho` from one eigendecomposition and a table of `c_b(d)`) can supply a predicted error
standard deviation to calibrate schemes that have no measurement path (ASH, E8).

## Format stability

Whatever the refactoring, the bytes written by the formats must not change: only random ids (the UUID of an ASH
projection matrix) and float differences within a tiny relative margin are tolerated. This is checked after every step
by `.scratch_golden/compare.sh`, see its README: 36 scenarios built with the pristine base commit and with the current
code, compared by exact digest and, as a diagnostic, by file with a float-noise classifier that cannot mistake an
off-by-one integer for noise.

## 5. Migration order, each step gated

Gate for every step (strict, decided by the owner):
(a) golden-score tests (`CanonicalScoreFormTests` plus per-scheme golden files) unchanged;
(b) search results deterministic and identical to baseline (same top-k, same scores) for every scheme, not just equal recall;
(c) QPS / latency / indexing / merge time not worse than baseline: A/B runs interleaved, one benchmark at a time, a
    baseline noise band measured up front and the candidate's median required to be at or above the baseline median
    (a step that lands inside the noise band but below the median is investigated, not waved through);
(d) for OSQ, segment bytes identical to baseline wherever the layout is not being changed on purpose.

1. **Baseline**: freeze recall / QPS / size / merge time for OSQ, ASH, E8, and golden scores.
2. **Visitor base**: extract the shared posting-visitor code (doc-id decode, mask filtering, `collectBulk`, sliced
   ordinals, byte-to-float widening). Pure refactor.
3. **Scheme skeleton**: generic Format/Reader/Writer + `QuantizerScheme` registry (no change to the meta bytes); OSQ and ASH become the
   first two schemes with their current layouts. Fold the mapper / `DiskBBQPlugin` `if` chains into the registry.
4. **Query side**: `QueryEncoder` / `PreparedQuery` for OSQ and ASH.
5. **Layout and kernel**: ASH onto canonical lanes (little endian, `o` folded, 12 B/vector) and the fused fold; ASH
   should get faster, OSQ untouched.
6. **Projection seam**: `LatentProjection` with merge inheritance; ASH reader implements `VectorPreconditioner` where
   a space rotation applies.
7. **Calibration and cost hooks** on the scheme; extend auto-calibration to ASH.
8. **Port E8** (PoC) as a third scheme to prove the seam, then delete the duplicated code.

## 6. Decisions and open questions

Decided:
* Regression bar is strict (see the gate above).
* `SegmentCalibrationParameters` stays OSQ-only in this iteration; ASH (and E8) are a second iteration.
* Gain: schemes keep today's semantics, nothing changes. ASH already stores the paper's stored-norm gain
  (`scale = ||x-c|| / ||code||`). OSQ is not a gain-shape code: it stores a per-vector optimized `[lower, upper]`
  interval, so lane `s` means `(upper - lower) * bitScale` there. Lane meaning is scheme defined.
* No new native code is needed. OSQ keeps its existing native/Panama fold (`diskbbq_apply_corrections_*_bulk`, fixed
  `lower/upper/sum/additional` layout) untouched. ASH is Panama/Java only today (the native factory delegates it to
  the Panama one), so its fused fold is written in Panama Java. Native ASH kernels would be an optional later
  optimization, not a prerequisite.

Open:
* Cost model (`diskbbq-cost-model` worktree): lanes and code bytes per vector come from `CostProfile`.
