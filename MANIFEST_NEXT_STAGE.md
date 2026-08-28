# Overlay manifest

Extract this archive at the repository root. Existing files are overwritten only for:

```text
pyproject.toml
src/region_guided_reranking_study/__init__.py
```

New files:

```text
README_NEXT_STAGE.md
src/region_guided_reranking_study/target_region_screening.py
src/region_guided_reranking_study/screening_research.py
scripts/run_screening_mechanism_study.py
scripts/run_screening_sequential_study.py
scripts/run_screening_drift_study.py
scripts/analyze_screening_studies.py
scripts/run_all_screening_studies.py
configs/region_screening_quick.json
configs/region_screening_full.json
tests/test_target_region_screening.py
tests/test_screening_research.py
```

Base repository commit used when preparing the overlay:

```text
bdbd64e9abc28385508b6f38ed2e2c697f5a0b1d
```
