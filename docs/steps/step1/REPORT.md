# R142-FP-11 第一阶段实验报告

## 结论

预注册判定为 `SUPPORTED_STAGE1_NO_VLA_CLAIM`，`accepted=true`。

在 `ForkPush2D-v1` 的 400 个 paired evaluation seeds 上，label-free
bottleneck detector 对真实分叉点的 median absolute error 为 0，且 100%
预测在误差 1 步以内。固定 32-candidate 预算下，proposed success@N 为
1.000，uniform split 为 0.440，random split 为 0.350；paired bootstrap
增益分别为 +0.560（95% CI [0.510, 0.6075]）和 +0.650（95% CI
[0.6025, 0.695]），两个比较均在 10/10 seed blocks 上胜出。

这只支持 synthetic 2D benchmark 中的最小机制，不支持 VLA、learned
policy 或真实机器人性能结论。按照原实验规划，通过后下一步应是小型
BC/diffusion-policy 仿真，而不是直接进入 VLA。

## Benchmark definition

`ForkPush2D-v1` 是对称的 2D manipulation task。候选在早期沿同一个
stem 移动，在 action step 4、5 或 6 的随机真实 bottleneck 选择 upper
或 lower gate 绕过中央障碍。两条成功 mode 的路径、终点和 reward 对称；
中心路线碰撞失败。

普通 candidate generator 共享 family-level selector。82% 的 episode
将整个候选族压到中心失败 basin，从而产生 candidate collapse。环境真值
由 prefix-preserving signed intervention 是否能首次到达两个不同成功
gate basin 定义，与候选标签独立保存。

Detector 只读取同一步的 state/action/latent prefix disagreement，不读取
final success、mode、failure reason、terminal score、future suffix 或
oracle truth。B0 中 97.25% episode 通过预注册 collapse validity gate
（要求至少 80%）。

## Candidate genealogy

每个 terminal candidate 保存：

- `candidate_id`, `parent_id`, `generation_step`, `split_action_step`；
- 完整 `state/action/latent` sequence 与 prefixes；
- RNG seed、terminal score、final success、final mode、failure reason。

正式 genealogy 分为 8 个 `genealogy.jsonl.gz` shard，总计 400 episodes、
10 policies；每个 proposed episode 包含 8 个 scout parents 和 24 个 local
children，总 terminal candidate budget 仍是 32。

![Candidate genealogy](../evidence/formal_pai/r3/figures/candidate_genealogy.png)

## Bottleneck detection

对 scout candidates 计算每步 median pairwise disagreement `D(t)`，再以
`median(ΔD) + 3*MAD(ΔD)` 为 robust threshold，取第一个同时满足增量阈值
和 1.5x ratio 的 action step。正式结果：MAE=0、median=0、
`P(error<=1)=1.0`。

![Bottleneck detection](../evidence/formal_pai/r3/figures/bottleneck_detection.png)

## Quantitative table

| Method | Candidate budget | success@N | mode discovery | successful modes/sample | localization median |
|---|---:|---:|---:|---:|---:|
| B0 best-of-N | 32 | 0.1550 | 0.0775 | 0.004844 | n/a |
| B1 uniform split | 32 | 0.4400 | 0.29375 | 0.018359 | n/a |
| B2 random split | 32 | 0.3500 | 0.2275 | 0.014219 | n/a |
| Proposed bottleneck-local | 32 | **1.0000** | **0.9225** | **0.057656** | **0** |
| A no detection | 32 | 0.4400 | 0.3625 | 0.022656 | 1 |
| A wrong early (`t*-2`) | 32 | 0.1550 | 0.0775 | 0.004844 | 2 |
| A wrong late (`t*+2`) | 32 | 0.1550 | 0.0775 | 0.004844 | 2 |
| A correct + random operator | 32 | 1.0000 | 0.98875 | 0.061797 | 0 |
| A full resampling at correct step | 32 | 1.0000 | 1.0000 | 0.0625 | 0 |
| A more samples B0 | 64 | 0.1550 | 0.0775 | 0.002422 | n/a |

![Quantitative results](../evidence/formal_pai/r3/figures/quantitative_results.png)

## Required ablations and failure cases

全部实验均已运行；没有因中间 gate 失败而停止其他实验。

- No detection：success@N 0.440，固定中点只能偶尔命中真实 bottleneck。
- Random split：0.350；命中真实 step 时 success=1.000，未命中时
  0.1586。
- Wrong location：early/late 均退回 0.155，与 B0 相同。
- More samples：64 candidates 仍为 0.155，说明 family-level correlation
  使 2x N 无法修复 collapse。
- Full resampling：正确位置达到 1.000，表明关键是 location，而不是某个
  特定 structured operator。

主要 failure case 是候选族在中心 basin 高度集中：B0、wrong-early 和
wrong-late 都在障碍中心碰撞。Uniform/random 的失败集中在 split schedule
未命中真实 `t*` 的 episode。12 个代表性 paired cases 已保存到
`evidence/formal_pai/r3/results/failure_cases.json`。

![Candidate trajectories](../evidence/formal_pai/r3/figures/candidate_trajectories.png)

## PAI execution evidence

- Formal replacement run: `r142-fp11-stage1-eval-20260823-1412-r3`
- JobId: `dlc1e1wg0af86rlq`
- Terminal PAI status: `Succeeded`
- Actual idle placement: `UseOversoldResource=true`
- Shape: 1 worker, 8 A800, 92 CPU, 1600Gi memory/shared memory
- Fault policy: Sync OnFailure, maximum 50 platform restarts, one launcher
  attempt per incarnation
- Persistent first work: completed shard-07, owner `2254:2254`
- Complete evaluation: 8/8 shards, 400 episodes, owner `2254:2254`
- PAI probe created: no

Predecessor r2 failed before first work because its launcher read the wrong
controller environment names. It was explicitly stopped, r3 repaired the exact
mapping, and after r3 first work and `Succeeded`, the r2 PAI service row was
deleted with `verified_absent=true`; registry, CPFS, placement, and error
evidence remain preserved.

## Decision boundary

The experiment passes the toy-mechanism gate because collapse is present,
localization predicts the true branching step, and proposed stably exceeds both
fixed-budget location baselines. It does **not** justify entering VLA directly.
No VLA experiment was run in this stage.
