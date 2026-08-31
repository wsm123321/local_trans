# v1-R Quick Attempt 2 Failed Before Acceptance

The experiment finished numerical generation and truth reveal, but the strengthened audit rejected the output with:

```text
candidate hash mismatch for development__11001__scale_0.7
```

Diagnosis: the CSV was written with `%.17g`, but pandas' default float parser did not guarantee exact round-trip recovery. The maximum parser difference relative to `float_precision='round_trip'` was approximately `8.88e-16`; the round-trip parser reproduced the stored candidate hash exactly.

Fix: every frozen/predecision CSV reload and analysis CSV load that participates in hash or exact decision reconstruction now uses `float_precision='round_trip'`. This failed directory is retained and is not merged with subsequent results.
