# M2D-6 design: consume the boundary disjoint proof (phase 2 of design 22b395d)

## The defect (verified against encoder.cpp at main eae4cf0d33)

The M2 stall map shows QmmVecQ4ResNorm -> QmmVecQ4Word as the #1 stall edge:
7,208 us/tok, 100% disjoint buffers, 100% barrier emitted. The two kernels
touch disjoint byte ranges (the ResNorm outputs feed a staging path disjoint
from the cache Word reads), yet the barrier fires on every occurrence.

Root cause by inspection: the false positive is not in DependencyTracker or
wave_levels (both exact, unit-tested). It is the batch-boundary path:

- record_dependency_barrier() ends with tracker_.clear(); head_synced_ = true.
- reset_dependency_tracking() sets head_synced_ = false.
- The batch head runs: if (!head_synced_ || tracker_.needs_barrier(...)) it
  records a full dependency barrier against the previous batch, regardless
  of ranges.

In the decode loop the graph chunking closes a submission between ResNorm and
Word every layer, so this edge is a fresh batch head every token: the
boundary barrier fires 100% of the time even though the head ranges are
provably disjoint from everything the closed batch touched.

The proof already exists host-side: design 22b395d's export (DepRecord:
disjoint + usc_changed + seq, computed from the GATED_BARRIERS tracker,
stored host-side, not consumed yet - the driver learns nothing until
phase 2). The disjoint bit for the head dispatch is computed by the exact
same tracker_.needs_barrier run, inverted.

## The fix (phase 2: consume the export)

1. Keep the tracker across the submission boundary: remove
   tracker_.clear() + the unconditional head barrier from
   record_dependency_barrier()'s tail.
2. At the batch head, run the standard needs_barrier(head_rd, head_wr)
   against the CARRIED tracker: disjoint -> skip the boundary barrier (safe:
   same-queue submissions overlap only where ranges overlap; provably
   disjoint ranges may run concurrently); overlap -> emit the boundary
   barrier and clear the tracker (everything before is synced by a real
   barrier), then add the head ranges.
3. Bounding: the tracker is a merged-interval set per buffer (O(log n)
   overlap queries); in the steady decode loop the working set is the
   weight + cache slices, so the merged intervals converge to the working
   set and stay bounded. When the tracked dispatch count exceeds a budget
   (e.g., 4096), emit a real boundary barrier and clear - the amortized
   fallback keeps memory bounded and correctness identical.

## Failing-first tests (test_runtime.cpp, boundary suite)

T1: a closed batch writing b[0,64), a fresh head reading c[0,64) (disjoint
buffer) - the boundary decision must be SKIP (pre-fix: barrier).
T2: a closed batch writing b[0,64), a fresh head reading b[0,64) - the
boundary decision must be BARRIER (real RAW).
T3: the carried set after a SKIP contains both the closed batch's ranges and
the head's (later overlaps still caught).
T4: the carried set after a real boundary BARRIER is cleared down to the
head's ranges (no stale history inflating later barriers).
T5: the budget fallback - crossing the dispatch budget emits a real boundary
barrier and clears (the amortized path), correctness identical.

T1-T4 are host-side pure logic once the boundary decision is a testable
function (boundary_decision(carried, head_rd, head_wr) -> enum); T5 needs
the budget constant surfaced the same way.

## After the fix lands

- The M2 stall map re-run: the ResNorm->Word edge barrier count drops from
  100% to ~0 on disjoint occurrences; the 7,208 us/tok stall edge is the
  recovery target.
- Determinism: 2048-token decode, identical greedy ids before/after.
- 5-pair ABBAs on the M2: 2B, 4B, 9B - barrier-count drop + matched tok/s.
- Cross-model exact-head GREEN.
