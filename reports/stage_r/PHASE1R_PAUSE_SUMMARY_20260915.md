# R142-FP-11 Stage-R / Phase-1R 阶段暂停总结

**状态：按用户要求暂停，不再创建或提交新的 Phase-1R Job。**

本文件记录截至 2026-09-15 的已验证成果、未完成边界和暂停原因。它不是
Phase-1R 完成证明，也不把运行中的日志、`FIRST_WORK` 或 partial artifact
当作科学 outcome。

## 冻结对象

- scientific commit：`10308c471a846f8636cf05e5a40a2dad64f4d8ec`
- frozen mapping：A0 ranks 0--3、A1 ranks 4--7、B0 ranks 8--11、B1 ranks 12--15
- frozen task/seed/budget/selection/threshold/checkpoint/statistical protocol：未修改
- canonical source：`/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R142-FP-11-Bottleneck-Local-Candidate-Split`
- frozen launcher：`pai/run_stage_r_phase1r_idle_4gpu.sh`
- launcher SHA-256：`9acf021472fa4735cb966fefb5e86f38eafc39e42d5ca6ff2d3c322fa4b3c483`

## 已完成的科学结果

### Phase-0R（完整、已封存）

- 40 个 LIBERO natural tasks，16 个初始状态/任务，32 个候选/状态。
- 共 20,480 次 full rollouts、658,349 次 policy forward、3,251,756 个环境步。
- 40/40 NPZ 与 metadata 通过协议、owner、覆盖范围和 SHA 校验。
- 14 个任务达到 `rho >= 3`，30 个任务有至少两个稳定成功模式，但 0 个任务满足冻结的联合保留门槛。
- `CHECKPOINT_1_STOP`，`retained_tasks=[]`；原始冻结协议下 Phase-1R 未获授权。
- RoboTwin 保持 `SOURCE_LIMITATION_UNVERIFIABLE`，没有替换策略、伪造轨迹或 outcome 后选样。

主要报告：`reports/stage_r/PHASE0R_REPORT.md`、
`reports/stage_r/PLAN_COMPLETION_AUDIT.md`。

### 本地 CPU smoke（仅机制验证，不是 VLA 证据）

现有 smoke artifact 的冻结判定为 `IDEA_FAILED_DO_NOT_ENTER_VLA`：

- proposed bottleneck-local：`success_at_n=1.0`、`candidate_success_rate=0.39453125`；
- B1 uniform split：`success_at_n=0.625`；
- B2 random split：`success_at_n=0.375`；
- localization gate 通过，但稳定增益与比较 gate 未通过；`accepted=false`。

该结果只说明 toy 2D benchmark 中的受控机制行为，不能外推到 learned policy 或
VLA。对应文件：
`outputs/cpu_smoke/aggregate/{summary.json,gate_decision.json,mechanism_diagnostics.json}`。

## Phase-1R 运行尝试与终止边界

r25 的四个 Job 已在 2026-09-07 07:30（北京时间停机窗口）停止并 exact
读回终态：

| shard | JobId | 终态 |
|---|---|---|
| A0 | `dlc15jqbukzjwji9` | Stopped / StoppedByUser |
| A1 | `dlcb5543u629snt4` | Stopped / StoppedByUser |
| B0 | `dlcj6tc7e074lhb2` | Stopped / StoppedByUser |
| B1 | `dlcr8hkayffscs29` | Stopped / StoppedByUser |

停机 receipt：
`/mnt/cpfs/zbl-cpfs-new/SHARE/leon/r142_fp11_stage_r/phase1r_monitoring/r25_stop_20260907T0730CST`

该 receipt 的状态为 `STOP_STATUS=COMPLETE`，双层 SHA/fsync 通过；停机时累计
991/9600 natural cells、0/40 完整 task、0 个顶层完整 shard。四个 r25 Job
不可重启、不可复用 partial；r26 未创建，当前没有新的科学 outcome。

## 为什么没有继续提交

1. 全局 controller audit 仍为 `17 failed / 20 passed / 320 subtests`，失败来自
   R16P14 模板。R142 scoped compatibility audit 虽通过，但明确
   `global_audit_override=false`、`submission_authorized=false`。
2. 2026-09-07 曾有原始 DLC 输出泄露 W&B secret。该凭据按已泄露处理；新 Job
   前必须由用户通过隐藏 canonical 输入 `./bin/pai-wandb-setup --replace`
   更换，并重新通过 entity `chen_jian-cj-workspace` 的 write-membership gate。
3. registry/CKPT 持久化的独立 fsync 没有形成可接受的通过证据；任何新叶写入
   前都必须 fail-closed。
4. canonical controller 相关文件存在并行修改，不能 reset 或盲目重绑定 wrapper。

因此暂停是安全和可复现性要求，不是把某个 `Running`、日志或 partial artifact
误判为完成。阶段恢复前仍需重新核对源码、冻结 commit、DLC/helper SHA、registry
与 CKPT fsync、credentials/W&B、0 旧 GPU，再按 outcome-blind structural clone
建立新 lineage；不得复用 r25。

## 暂停后的结论

- 已发布并保留：Phase-0R 完整负结果、CPU toy smoke 的失败 gate、r25 停机证据、
  controller/凭据阻塞证据。
- 未完成且不应声称完成：Phase-1R 四个完整 shard、每任务 240 cells 的完整结果、
  `COMPLETED_EVALUATION_RESULT.json`、最终合并、冻结 Phase-1R analysis、
  `CHECKPOINT_2` 以及任何 Phase-2R。
- 本阶段在用户明确要求下停在当前证据边界；后续若恢复，必须从全门槛重新开始，
  不得 outcome-select、替换或混用运行。
