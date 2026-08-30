# Rejected Quick Pilot attempt 1

- Stage: `local-surrogate-transfer-pilot-v1-quick`
- Frozen implementation commit: `45944c08f237b223ee1d70fc69d73320a8b60154`
- Status: **rejected before analysis**
- Failure point: post-experiment manifest construction
- Error: output artifact paths supplied on the command line were relative paths, while `_write_manifest` called `Path.relative_to(REPO_ROOT)` without first resolving them. All per-instance computations completed and four CSV files were written, but the manifest, analysis, and audit were not produced.
- Scientific status: these CSV files are incomplete workflow artifacts and are not used, merged, summarized, or compared with subsequent attempts.
- Corrective action: normalize configuration and output paths against `REPO_ROOT` before directory creation and downstream subprocess invocation; rerun the complete Quick workflow from scratch in a new output directory.
