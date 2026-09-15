# R142-FP-11 总结与阶段性收尾

**仓库**：`mikasaTu/R142-FP-11-Bottleneck-Local-Candidate-Split`  
**收尾日期**：2026-09-15（Asia/Shanghai）  
**收尾状态**：本阶段停止，不再提交新的 PAI 任务；保留全部代码、计划、报告、运行账本和失败证据。

## 1. 原始 idea 假设

本项目要验证的不是“更多 candidate 是否提升成功率”，而是：当 candidate sampling 出现 mode collapse、候选共享同一个错误决策前缀时，是否存在一个 earliest bottleneck state；在这个位置做局部 candidate split，是否比相同计算预算下的 uniform/random 增加采样更有效。

完整原始假设和顶层计划见：

- `docs/steps/context/ORIGINAL_IDEA.md`
- `docs/steps/context/EXPERIMENT_PLAN.md`
- `docs/steps/context/ORIGINAL_IDEA.xml`
- `docs/steps/context/EXPERIMENT_PLAN.xml`

## 2. 各阶段结论

### Step 1：Toy mechanism validation

基于 `ForkPush2D-v1` 的 400 个 paired seeds，验证了人工设计的最小机制：候选族存在共享失败前缀，label-free disagreement detector 能定位真实分叉位置。

- detector median absolute error：`0`
- `P(error <= 1)`：`1.0`
- proposed success@32：`1.000`
- uniform split：`0.440`
- random split：`0.350`
- 所有预注册 ablation 均执行，包括 no-detection、wrong-location、more-samples 和 full-resampling。

结论：`SUPPORTED_STAGE1_NO_VLA_CLAIM`。这只支持 synthetic 2D toy mechanism，不支持 learned policy、VLA 或真实机器人结论。

报告：`docs/steps/step1/REPORT.md`、`reports/STAGE1_EXPERIMENT_REPORT.md`。

### Step 2 / Stage-2A：独立 learned-policy 证伪

在未修改的官方 LeRobot Diffusion PushT 上进行 descendant-blind natural snapshot、真实 DDPM genealogy、calibration/held-out suffix 和固定 sample-NFE 对比。

- natural location-sensitive snapshots：`0/24`
- natural hard recoverability cliffs：`0/8`
- 最大 location spread range：`0.040996`（预注册门槛 `0.10`）
- oracle-local 相对 always-early：`+0.002258`，但未超过 random/uniform，也远低于门槛
- `always-early` 与 oracle 等价：`true`

结论：`R142_FP11_CORE_HYPOTHESIS_WEAKENED`；按预注册停止规则，不进入 VLA/π0 扩展。

机制反解：latent disagreement 的收缩没有转化为 executed-action/outcome support 的收缩；许多 latent 变化落在行为不敏感方向，且固定 NFE 下晚分叉会牺牲可用 suffix 数。详见 `reports/STAGE2A_MECHANISM_REVERSE_EXPLANATION.md`。

报告和原始结果：

- `docs/steps/step2/REPORT.md`
- `reports/STAGE2A_EXPERIMENT_REPORT.md`
- `results/stage2a_formal/`
- `results/stage2a_continuation/`

### Step 3 / Stage-R：trajectory-axis revalidation

使用固定 pi0.5-LIBERO 运行 40 个任务、20,480 个 eventual-termination rollouts，并按冻结规则寻找共享失败前缀、真实 trajectory split 和稳定 mode。

- 所有 40 个任务均分析完成
- natural initial-state families 中 `p_e <= 1/32` 的比例为 `0`
- all-fail family 数量为 `0`
- `t_div` 对 natural families 均 undefined
- 稳定成功 mode 存在，但与 shared failing-prefix 不共现
- RoboTwin：`SOURCE_LIMITATION_UNVERIFIABLE`，未引入替代轨迹

结论：`CHECKPOINT_1_STOP`，`phase1_authorized=false`。这是 precondition failure，不是“所有策略都不存在 bottleneck”的普遍性结论。

报告：

- `docs/steps/step3/REPORT.md`
- `reports/stage_r/PHASE0R_REPORT.md`
- `reports/stage_r/PLAN_COMPLETION_AUDIT.md`
- `results/stage_r/`

### Step 4 / Stage-S：substrate qualification screen

Step 4 的计划、脚本、运行账本和失败证据已经归档；本阶段没有伪造科学结果。

已确认：

- A asset preflight 曾完成可接受的 asset lineage，但尚未形成可启动 main screen 的完整 frozen protocol。
- C under-trained PAI pipeline 成功结束，checkpoint、RNG sidecars 和 SHA manifest 完整。
- C calibration 在 episode 之前 fail-closed：固定 `CleanPi05LiberoPolicy` 要求 OpenPI `params/` tree，而 C 训练产物是原生 `model.safetensors`。
- 因此没有 C pooled-success、冻结 calibration report、`FROZEN_PROTOCOL.json`、A/B/C main、S4 或 S5 科学结果。

结论：`CHECKPOINT_BLOCKED_ON_C_SERIALIZATION`。这是 checkpoint/interface serialization blocker，不是 success rate 或 substrate quality 的科学结论。禁止无授权地转换 checkpoint、插入伪造 `params/` 或改变 frozen substrate。

报告和证据：

- `docs/steps/step4/PLAN.md`
- `stage-s/STEP4_STATUS.md`
- `stage-s/CALIBRATION_REPORT.md`
- `stage-s/results/calibration/C/CALIBRATION_FAILURE.json`
- `stage-s/results/validation/MAIN_VALIDATION_BLOCKERS.json`
- `stage-s/operations/`

## 3. 代码、测试和可复现性

仓库保留：

- Step 1 toy benchmark、candidate genealogy、bottleneck detector、baseline/ablation 和绘图脚本；
- Step 2A 官方 learned-policy tracing、snapshot/replay、genealogy、固定-NFE 分析和机制反解；
- Stage-R replay integrity、Phase-0R merge/finalize/analyze、PAI runbooks 和 failure records；
- Stage-S LIBERO/RoboTwin runtime、calibration/main/S4/S5 launcher、checkpoint acceptance、asset preflight、freeze validator 和完整运行账本；
- Feishu 计划/报告 XML 快照以及 GitHub 发布证据。

已记录的验证包括 Python compile、JSON parse、shell syntax、`git diff --check` 以及 Stage-S 定向测试（C calibration、B/C main、S4/S5 等共 25 tests passed 的阻塞验证批次）。Queued/Running/FIRST_WORK/partial shard 均没有被当作科学完成。

## 4. 机制总览

Step 1 中，人工构造的 shared prefix 和中间 gate 使 bottleneck-local split 直接命中真正的 failure branching point，因此效果显著。Step 2A 和 Step 3 显示，在真实 learned-policy / 固定 LIBERO 分布中，latent 或 initial-state heterogeneity 并不等价于 shared failing-prefix；没有候选族共同失败，local split 就没有可作用的分叉点。Step 4 当前停在接口层，尚未产生可用于机制判断的 substrate screen 数据。

因此，现阶段最强的可证伪结论是：R142-FP-11 的最小 toy 机制成立，但在自然 learned-policy 分布上尚未被观察到；不能把 Step 1 的正结果外推到 VLA。

## 5. 发布和停止边界

- 本总结与此前所有阶段材料一起发布到 GitHub `main`。
- 本阶段停止，不删除任何 PAI/CPFS 失败证据，不删除已完成结果，不再提交新的 PAI 任务。
- 若未来重启 Step 4，必须先由用户明确授权保持权重与 lineage 不变的 native checkpoint/interface repair，再重新完成 C calibration 和冻结协议；本文件不授权该修复。

