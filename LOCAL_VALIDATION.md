# Local Validation Record

This record documents code validation performed before packaging. It is not a scientific result.

## Unit tests

```text
7 passed
```

Covered behaviors:

- affine response-scale invariance of extracted rank structure;
- higher score near a known local optimum;
- inclusion of non-elite boundary observations;
- recovery of two separated elite basins;
- component matrix and region-index interfaces;
- finite controlled-recovery metrics;
- GP local-surrogate execution path.

## Experiment smoke tests

- The controlled recovery quick configuration completed all eight configured instances and produced 24 method rows with no extraction exception.
- The held-out validation and analysis pipeline completed against a temporary API-compatible landscape stub used only to test script contracts.
- Temporary smoke-test landscapes and generated results are not included in the overlay.

Full scientific conclusions must be based on the repository's actual benchmark landscapes and the frozen full configuration.
