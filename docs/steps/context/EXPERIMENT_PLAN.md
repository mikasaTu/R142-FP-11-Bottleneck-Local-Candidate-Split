# 实验规划

## Phase 0：最小 toy 验证（必须先做）

目的：

验证机制本身是否成立。

不要马上上 VLA。

因为如果：

> bottleneck detection 本身不能预测 candidate collapse

那么后面全部没有意义。

---

## Phase 1：小型 BC / diffusion policy 仿真验证

目的：

证明：

```Plain Text
Bottleneck-local split
>
uniform resampling
```

在相同计算预算下有效。

---

## Phase 2：VLA / World Model 验证

根据 Phase 1 结果选择。

如果证明：

-  candidate genealogy 有意义；
-  bottleneck 可检测；

再上：

-  π0.5
-  OpenVLA
-  SmolVLA
-  World Model planner
