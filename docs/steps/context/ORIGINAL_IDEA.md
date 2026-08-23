<title>R142-FP-11-Bottleneck-Local-Candidate-Split</title>

# 最终验证的核心 hypothesis（必须固定）

不要验证：

> “增加 candidate diversity 是否提升成功率”

这个已经被大量方法做过。

真正验证：

> **当 candidate sampling 出现 mode collapse（所有候选共享同一个错误决策前缀）时，是否存在一个 earliest bottleneck state，在该位置进行局部 candidate split，比同计算预算的随机增加采样更有效？**

核心不是：

```Plain Text
更多 sample
```

而是：

```Plain Text
找到候选族共同失败的最早分叉点
↓
只在那里增加 diversity
↓
减少无效 sampling
```
