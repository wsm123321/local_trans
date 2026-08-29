# Source Structure Stage Overlay

Base commit inspected: `917e38a32c947d9394d7d3ff96cbbe39c6236bd0`

## Added

- `src/region_guided_reranking_study/source_local_structure.py`
- `src/region_guided_reranking_study/source_structure_research.py`
- `scripts/run_source_structure_recovery.py`
- `scripts/run_source_structure_validation.py`
- `scripts/analyze_source_structure_study.py`
- `scripts/run_all_source_structure_studies.py`
- `configs/source_structure_quick.json`
- `configs/source_structure_full.json`
- `tests/test_source_local_structure.py`
- `README_SOURCE_STRUCTURE_STAGE.md`
- `PROTOCOL_SOURCE_STRUCTURE.md`
- `MANIFEST_SOURCE_STRUCTURE_STAGE.md`

## Replaced

- `src/region_guided_reranking_study/__init__.py`
- `pyproject.toml`

## Local validation

- New unit tests: 7 passed.
- Controlled recovery quick smoke test completed with no extraction failures.
- Held-out validation and analysis scripts completed against a temporary API-compatible landscape stub; the stub is not included in the overlay.
