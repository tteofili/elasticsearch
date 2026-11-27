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

## 7. Generic encoders and decoders

Goal: code that is written once for every scheme (laying out and writing blocks, characterizing a quantizer, a reference
scorer, calibration) sees an *encoder* and a *decoder*, and never OSQ, ASH or the scheme to come.

### 7.1 Feasibility: shown, for the encode and decode halves

Both schemes already reduce to the same two halves, and a prototype exists that runs through them:

* **Encode.** A block of vectors that share a posting-list centroid becomes, per vector, a code (one level per latent
  coordinate for OSQ and ASH) and a handful of numbers. OSQ gets it from `OptimizedScalarQuantizer.scalarQuantize` (one
  vector at a time), ASH from `AsymmetricHashingQuantizer.BlockEncoder` (a block projected with one matrix multiply, float
  or int8 as the segment writer does).
* **Decode.** The reconstruction is affine, `lower + scale * level`, in both: OSQ with `lower = lowerInterval` and
  `scale = (upper - lower) / (2^b - 1)`, ASH with `scale` and `lower = -scale * (2^b - 1) / 2` (its codes are centered).
  A codebook that is not a grid (E8) supplies its own decoder.

Prototype, in the worktree (`diskbbq/quantize`, `ash/AshDocEncoder`, `diskbbq/calibrate/EncoderDistortionRateQuantizer`):

| piece | role |
|---|---|
| `DocEncoder` | block-shaped: `reset(centroid)`, `add(vector)`, `encode()`, `read(i, levels, lanes)`, `latent(i, out)` |
| `DocLanes` | what is stored beside a code, in the same terms for every scheme: `scale`, `lower`, `sum`, `offset` |
| `DocDecoder` / `AffineDocDecoder` | `(levels, lanes) -> latent reconstruction` |
| `OsqDocEncoder`, `AshDocEncoder` | adapters over the quantizers segment writers already use |
| `EncoderDistortionRateQuantizer` | a `DistortionRateQuantizer` whose `reconstruct` is *encode, decode, compare*, for any encoder |

Evidence: before the swap, the generic reconstruction was compared with the two hand-written ones on 600 residuals each
(OSQ at 1, 2 and 4 bits, lambda 0.1 and 1.0, with and without rotation; ASH at 1, 2 and 4 bits; including zero-norm
residuals): agreement to 1e-4. Both `OsqDistortionRateQuantizer` and `AshDistortionRateQuantizer` now extend the generic
class and lost their private copies, so what is characterized is what an encoder writes and cannot drift from it. The
recall estimator still agrees with measured recall as before (0.83 / 0.33 points of mean absolute error at eta 1 / 3, rotated
configurations), the format gate is unchanged (36/36 digests, 608 files identical) and `DocEncoderTests` runs the same
contract checks (levels in range, `sum` lane, decoding close to the direction, blocks reusable) against both schemes through
the interface alone, 30 iterations.

### 7.2 The layers

```
LatentProjection      D -> d            OSQ: identity (+ the Preconditioner outside the codec); ASH: learned W; E8: identity
DocEncoder            block of vectors  -> code symbols + DocLanes           (scheme specific; writers use it)
DocDecoder            code + lanes      -> latent reconstruction             (affine for OSQ/ASH; codebook lookup for E8)
QueryEncoder          query             -> PreparedQuery.forPostingList(c)   (OSQ: quantize q - c; ASH: project + quantize once)
ScoreKernel           PreparedQuery x block of codes -> scores               (native/Panama; stays scheme specific)
DistortionRateQuantizer   = LatentProjection + DocEncoder + DocDecoder + QueryEncoder, for the law (characterization)
```

What is generic and what stays with the scheme:

| Concern | Generic | Scheme specific |
|---|---|---|
| characterize a quantizer, estimate recall, isotropy check | yes (`EncoderDistortionRateQuantizer`, `RecallEstimator`) | encoder, decoder, projection |
| reference scorer (test oracle for kernels) | yes: score = `<q_latent, decode(code)> + <q, mu>` | none |
| block iteration, doc-id and mask handling | yes (`PostingBlocks`, already shared) | none |
| packing levels into bytes, on-disk block layout | the *framework* of a layout writer | the layout (`DiskBBQBulkWriter` bit-planes for OSQ, `ashPack` for ASH) |
| per-vector corrections on disk | none | native lanes: OSQ `(lower, upper, additional, sum)`, ASH `(scale, offset, sum, vecCentroidDot, vecCentroidSqDist)` |
| bulk scoring | the interface | the SIMD/native kernel |

Three things that look generic and are not, and are therefore deliberately *outside* the encoder interface:

1. **On-disk lanes are not `DocLanes`.** `DocLanes.scale` is derived (`(upper - lower) / (2^b - 1)` for OSQ), so rebuilding
   OSQ's stored `upper` from it can differ by an ulp, and the bytes would change. Writers keep writing each scheme's native
   corrections, from the scheme's own encoder; `DocLanes` is the canonical view for generic readers of an encoder.
2. **Packing and layout depend on the format, not on the scheme.** The encoder returns levels; a layout writer packs them. ASH
   packs from centered float codes today; levels map to them exactly (`code = level - (2^b - 1)/2`), to be proven by a test
   before the packers are shared.
3. **Kernels are the performance boundary** and stay as they are.

### 7.3 Plan

Gate on every step, as in section 5: (a) `compare.sh` (bytes identical), (b) identical top-k and scores, (c) no regression in
indexing, merge and search time (interleaved A/B, one benchmark at a time), (d) golden-score tests unchanged.

| # | Step | Status and what was learned |
|---|---|---|
| 0 | `DocEncoder`/`DocDecoder`/`DocLanes`, OSQ and ASH adapters, generic `reconstruct`, scheme contract tests | **Done.** Gate (a), (d); recall estimator numbers unchanged |
| 1 | Writers quantize through one place | **Done for OSQ.** `OsqVectorQuantizer` (`diskbbq/quantize`) is the single-vector, copy-free form of `OsqDocEncoder`: quantize against the list centroid and fix up the additional correction against the parent centroid, float and byte. It replaces the four copies in `ESNextDiskBBQVectorsWriter` (merge temp file loop, its overspill loop, `OnHeapQuantizedVectors`, `OnHeapQuantizedByteVectors`). Bytes identical (36/36 digests, 608 files). Timing of the exact code that changed, old inline sequence vs the extracted method, alternated on the same vectors (dim 768, 20k vectors, 41 reps, Euclidean and dot product, 1 and 4 bits, with and without a parent centroid): ratio new/old 0.976 to 1.0006, no regression. **Deliberately not done for ASH:** its list writer already uses the production `BlockEncoder` and needs its float codes for packing; going through the generic interface would only add a conversion per vector. `AshDocEncoder` serves the generic consumers |
| 2 | Packing and native lanes behind the encoder | **Done, and simpler than planned.** `ashPack` already rounds codes to levels and then packs the levels, so levels-to-bytes costs nothing for either scheme: `DocEncoder.pack` and `nativeLanes` return what the writers write, tested byte for byte against the OSQ writer's sequence (1, 2, 4 and 7 bits, padded dimensions, both similarities) and the ASH list writer's (`BlockEncoder` + `ashPack`, all five lanes). New public `ESVectorUtil.ashPackLevels`, with a test that pins `ashPack == ashPackLevels(levels)` |
| 3 | Query side behind `QueryEncoder`/`PreparedQuery` | **Done.** `OsqQueryEncoder` is the reader's `QueryQuantizer` extracted unchanged (LRU of 16, lazy `reset`/`quantizeQueryIfNecessary`, one-iteration quantizer); the reader constructs it. `AshQueryEncoder` shares the min-max grid arithmetic with `AshPostingsVisitor.quantizeQuery`. `queryError` of every `EncoderDistortionRateQuantizer` is now derived from `exactLatent`/`servedLatent` of the real query encoder instead of two hand-written copies (which used a different number of OSQ iterations than serving does). Search results identical (gate covers top-k and score bits) |
| 4 | Reference scorer | **Done, but not the one planned.** The decoder alone cannot reproduce a kernel: a score also needs each vector's own correction and per-list query scalars with scheme-defined meaning. So the canonical form of section 3 became code: `ScoreLanes` (from `DocEncoder.scoreLanes`), `QueryScalars` and `rawDot` (from `PreparedQuery`), and one `CanonicalScorer`. It reproduces the real `ES940OSQVectorsScorer.applyCorrectionsIndividually` (1, 2, 4 bits, dot product and Euclidean) and the verbatim ASH visitor formulas (2, 4 bits, both similarities, float and integer query) on codes made by the generic encoders; 30 iterations, no failures. The scalar oracle for kernels now exists; wiring each native/Panama kernel test to it is the follow-up |
| 5 | Port E8 as the third scheme | **Not possible in this worktree**: the E8 PoC lives in another worktree. `codeLength()` and a non-affine decoder are in the interface for it; this step is where that is proven |
| 6 | Optional, format-changing: canonical lanes and one layout for every scheme; ASH onto the fused fold | Not started; outside the byte-identity gate |

Risks and open points:

* Step 1 touches the hot loop of indexing. The extracted method was timed against the code it replaced (no regression, see
  the table), which covers the changed code and not a whole index or merge: the strict bar still asks for the interleaved
  A/B of indexing and merge time on real datasets before this is merged, which needs the disk and the exclusive machine.
* ASH's int8 projection (`bits <= MAX_INT8_PROJECTION_BITS`) means the latent the encoder keeps carries int8 rounding; the
  characterization sees what the writer sees, which is intended, but a scheme comparison at equal bits mixes that in.
* `codeLength()` and the decoder are enough for E8 on paper; step 5 is where that is proven rather than argued.
* Two float lanes of an encoder compared across runs can differ by an ulp (Vector API reductions sum in another order once
  C2-compiled), so tests that compare an encoder with a reference quantization compare codes exactly and floats to a few ulps.
