# Grounded Imagination 实验日志

---

## 第一轮实验：DMC cup_catch + manipulator_bring_ball

### 配置
- 日期：2026-04-06 启动
- 硬件：8×A100-SXM4-80GB
- 配置：dmc_vision, 1.1M steps, train_ratio=512
- Grounded：MoE (4 experts, Top-2) + TRD + 自适应 rollout 加权（无 DAgger, 无优先级采样）
- Baseline：原版 DreamerV3

### 中期结果 (2026-04-07, ~30% 进度)

| 实验 | 步数 | 分数 |
|------|------|------|
| Grounded cup_catch s0 | 336K | 923-964 (均值~949) |
| Grounded bring_ball s0 | 336K | ~0 |
| Grounded bring_ball s1 | 336K | ~0 |
| Grounded bring_ball s2 | 336K | 0 |
| Baseline cup_catch s0 | 640K | 963-992 (均值~973) |
| Baseline bring_ball s0 | 640K | ~0 |
| Baseline bring_ball s1 | 624K | 0 |
| Baseline bring_ball s2 | 640K | ~0 |

### 训练速度
- Baseline 约为 Grounded 的 2 倍（640K vs 336K 同时启动）
- 符合预期：MoE + TRD 额外计算开销约 17%，加上 JIT 编译更慢

### 关键发现

#### 发现 1：Grounded 在简单任务上分数低于 Baseline
- **现象**：cup_catch Grounded 均值约 949，Baseline 约 973，差距约 24 分
- **原因分析**：TRD 信任分数 real=0.66, fake=0.30
  - TRD 成功区分了真假转移（好现象）
  - 但 real 分数只有 0.66（不是接近 1.0）
  - 累积信任 = 0.66^15 ≈ 0.003，几乎把所有想象数据都降权为 0
  - 在 cup_catch 这种简单任务上，想象数据本来就很准，不该被大幅降权
- **结论**：**TRD 累积信任机制过于激进，在简单任务上过度惩罚了有效的想象数据**
- **对应 spec 风险**：风险表中"TRD 训练不稳定"和"审稿人认为三组件拼接"

#### 发现 2：bring_ball 两边都是 0
- **现象**：Grounded 和 Baseline 在 bring_ball 上分数都是 0
- **分析**：与 spec 中 2.1 节的实验证据一致（DreamerV3 原版最终分 0，最佳分 236）
- **结论**：30% 进度还太早，需等跑完看最终结果。但即使跑完，这个任务可能对两边都太难

#### 发现 3：TRD 学习状态
- trd_score_real: 0.66, trd_score_fake: 0.30
- TRD loss: 0.73（在下降）
- **结论**：TRD 能有效区分 posterior（真实校正）vs prior（模型预测），判别器本身工作正常

### 改进方向

#### 方向 1：缓解累积信任过度降权
问题核心：单步 0.66 不算低，但 cumprod 15 步后趋近 0。

**选项 A — 去掉累积，改为每步独立加权：**
```python
# 当前: weight *= cumprod(trust, axis=1)
# 改为: weight *= trust_per_step（不累积）
```
优点：简单，不会过度惩罚。缺点：失去"后续步建立在错误基础上"的累积效应。

**选项 B — 温度缩放 trust score：**
```python
trust_adjusted = trust ** temperature  # temperature < 1
# 例如 temperature=0.3: 0.66^0.3 ≈ 0.87
# 累积: 0.87^15 ≈ 0.12（仍有意义，不趋近 0）
```
优点：保留累积逻辑，可调参。缺点：多一个超参数。

**选项 C — 设置 trust 下限：**
```python
trust_clipped = max(trust, trust_floor)  # trust_floor=0.5
# 0.66 → 0.66, 0.3 → 0.5
# 累积: 0.66^15 ≈ 0.003 → 但下限保证不会太低
```

**倾向**：选项 B（温度缩放），保留框架完整性的同时可控。

#### 方向 2：改善 TRD 负样本质量
当前负样本 = softmax(prior_logit)，这是模型"未见观测时的预测"。
prior 和 posterior 的差异在简单任务上很小（模型已经很准了），但 TRD 还是能区分。
这说明 TRD 可能在学 prior vs posterior 的统计差异，而非真正的"想象质量"。
后续可考虑：用 batch shuffle 的负样本作为补充，或只在想象轨迹（而非 observe 数据）上训练 TRD。

---

## 待记录

### 第一轮最终结果
（等实验跑完后补充）

### 第二轮实验：MetaWorld 操作任务
（待启动）
