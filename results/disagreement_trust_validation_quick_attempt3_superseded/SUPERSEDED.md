# v1-R Quick Attempt 3 Superseded Before Analysis

This revised Quick run completed numerical generation and passed its strengthened run audit. It was not accepted as an analyzed result.

A second independent review found that the analyzer recorded the fraction of finite cluster-bootstrap replicates for common-support AUPRC contrasts, but the automatic verdict enforced that minimum only for accepted-harm risk. One-class or empty common-support resamples could therefore be silently omitted from an AUPRC confidence interval.

Fix before the next run:

- require the frozen minimum finite-bootstrap fraction for both primary AUPRC contrasts;
- classify insufficient finite AUPRC or harm bootstrap replicates as `inconclusive`, not `do_not_advance`;
- verify the current implementation hashes recorded in the run manifest;
- bind artifact verification to the exact supplied run directory;
- reconstruct thresholds and all accept/fallback decisions during the run audit.

This directory is retained for audit and is never merged with subsequent v1-R results.
