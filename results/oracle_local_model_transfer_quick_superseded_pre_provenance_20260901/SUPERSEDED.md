# Superseded Gate-0 Quick Artifact

本目录保留 2026-09-01 provenance 加固和决策规则修正之前的 Gate-0 quick 产物，仅用于审计历史。

不得将本目录中的 `promising_rank_transfer` 作为当前结论。该标签来自旧 analyzer 的任意 tie-break，未直接检验 Value/Dual 相对 Rank 的增量；旧 runner 也没有持久化足以独立重建 source-label permutation、target truth hash 和 seed lineage 的无损输入。

当前有效的 quick artifact 位于：

`../oracle_local_model_transfer_quick/`

当前机器判定为：

- `label = promising_value_or_dual_head_transfer`
- `selected_head = value`

本目录未被删除或覆盖，原始文件保留不变；本说明文件仅标记其证据状态。
