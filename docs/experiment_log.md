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

#### 方向 3（已实现）：折扣累积信任 — 引入 trust_gamma

**问题本质**：原始 cumprod 假设"一旦某步不可信，后续所有步永久不可信"。但世界模型
可能在错误后自我纠正回合理状态，不应永久惩罚。类比蒙特卡洛折扣系数 γ。

**数学形式**：
```
原始 cumprod:
  cumulative_t = trust_1 × trust_2 × ... × trust_t
  = cumprod(trust)

折扣累积 (递推):
  cumulative_t = trust_t × cumulative_{t-1}^γ

展开:
  cumulative_t = trust_t × trust_{t-1}^γ × trust_{t-2}^(γ²) × trust_{t-3}^(γ³) × ...

在 log 空间递推:
  log_cum_t = log(trust_t) + γ × log_cum_{t-1}
```

**γ 的作用**：
- γ=1.0 → 原始 cumprod（无折扣，当前问题）
- γ=0.0 → 只看当前步（独立加权）
- γ=0.5 → 折中，早期误差逐步遗忘

**收敛值** = trust^(1/(1-γ))，当 trust 恒定时：
```
trust=0.66 (简单任务):
  γ=1.0 → 0      (崩溃)
  γ=0.5 → 0.44   (合理)
  γ=0.3 → 0.55   (宽松)

trust=0.35 (接触区域):
  γ=0.5 → 0.12   (仍然明显低，保留检测能力)
```

**实现**：agent.py 中用 jax.lax.scan 在 log 空间递推，configs.yaml 新增 trust_gamma=0.5。

**优势**：
- 简单区域不崩溃（稳定在 0.44 而非 0.003）
- 接触区域仍能检测（降到 0.12）
- γ 可调，物理意义清晰（"误差记忆长度"）
- 与蒙特卡洛折扣因子类比，理论上可纳入 bound 分析

---

## 第二轮实验：MetaWorld 操作任务（proprio only）

### 配置
- 日期：2026-04-07 启动
- 硬件：8×A100
- 配置：metaworld (size50m), 1.1M steps, train_ratio=512, envs=16, proprio only
- Grounded：MoE + TRD + 折扣累积信任 gamma=0.5（无 DAgger, 无优先级采样，旧代码）
- Baseline：原版 DreamerV3

### 中期结果 (2026-04-08)

| 任务 | Grounded | Baseline | 差距 |
|------|----------|----------|------|
| reach (简单) | 4807 | 4833 | -26 (持平) |
| push (中等) | 4216 | 4748 | **-532** |
| door-open (中等) | 1799 | 4242 | **-2443** |
| pick-place (难) | 6.24 | 3030 | **-3024** |
| cup_catch (DMC) | 986 | 965 | +21 (持平) |
| bring_ball (DMC) | ~0 | ~0 | 持平 |

### 关键发现

#### 发现 4：Grounded 在操作任务上全面落后 Baseline

- **现象**：除 reach（最简单）持平外，push/door-open/pick-place 全部大幅落后
- **TRD 指标**：trd_score_real=0.65-0.69, trd_score_fake=0.28-0.31
  - TRD 能区分真假，但 real 分数仍然不够高
  - 折扣累积 (gamma=0.5) 后收敛值 ≈ 0.66^2 = 0.44
  - 想象数据被降权到 0.44 × 原始权重，严重削弱策略学习
- **根因确认**：TRD 信任加权机制在当前设计下是有害的
  - 它学到的是 posterior vs prior 的统计差异，不是"想象质量"
  - 即使世界模型很准，TRD 也给不了接近 1.0 的分数
  - 结果：好的想象数据也被降权 → 策略学不好

#### 发现 5：训练速度差异
- Baseline MetaWorld (proprio): fps/policy ≈ 28-30
- Grounded MetaWorld (proprio): fps/policy ≈ 10
- Grounded 慢 3 倍（MoE + TRD 计算开销）

### 诊断：问题在 MoE 还是 TRD？

需要分离实验：
- MoE only (moe=True, grounded=False)：如果恢复到 Baseline 水平或更好 → 问题在 TRD
- TRD only (moe=False, grounded=True)：如果仍然落后 → 确认 TRD 是罪魁祸首

### 下一步：MoE-only 实验

在 pick-place 和 door-open 上跑 MoE-only vs Baseline，确认 MoE 本身是否有害。

---

## 第三轮实验：MoE-only 消融

### 目的
确认性能下降来自 TRD 信任加权还是 MoE 架构本身。

### 配置
- MoE only: `--agent.dyn.rssm.moe True --agent.grounded.enabled False`
- Baseline2: 原版 DreamerV3（同代码版本重新跑，公平对比）
- 任务: reach, push, door-open, pick-place

### 中期结果 (2026-04-08, 服务器关闭前最后数据)

注: MoE-only 步数约 27K, Baseline2 约 74K, Grounded 约 144K, 步数不对齐。
Baseline (上一轮) 已跑完，步数最多。

| 任务 | MoE-only (27K) | Baseline2 (74K) | Grounded (144K) | Baseline 上轮 (完整) |
|------|---------------|-----------------|-----------------|---------------------|
| reach | 1704 | 4828 | 4807 | 4833 |
| push | 25 | 25 | 4216 | 4748 |
| door-open | 382 | 3863 | 685 | 4464 |
| pick-place | 5.6 | 8.5 | 6.2 | 3178 |

### 关键发现

#### 发现 6：MoE 本身也有害

- **现象**：MoE-only 在所有任务上都远低于 Baseline2
- **步数差异**：MoE-only 步数是 Baseline2 的 1/3（27K vs 74K），但即使考虑进度差异，
  reach（1704 vs 4828）和 door（382 vs 3863）的差距也无法仅用步数解释
- **训练速度**：MoE-only fps≈10, Baseline fps≈27, MoE 导致 3x 训练减速
- **可能原因**：
  1. 4 个专家分摊数据，每个专家训练不充分
  2. Router 早期随机分配，导致专家无法有效分化
  3. Balance loss 可能干扰主训练目标
  4. Top-2 routing 引入额外噪声

#### 发现 7：三组件全部无效

| 组件 | 预期效果 | 实际效果 | 原因分析 |
|------|---------|---------|---------|
| MoE Dynamics | 多专家建模不同动力学 | **有害** | 数据分散+router不稳定+3x减速 |
| TRD 信任加权 | 降权不可信想象 | **有害** | 度量的是posterior vs prior差异，不是想象质量 |
| DAgger | 主动纠正不准区域 | 未验证 | 依赖TRD信号，TRD本身有问题 |
| 优先级采样 | 聚焦不准区域训练 | 未验证 | 触发了Mixture selector __len__ bug |

### 反思

1. **假设未验证就开始实现**：假设"世界模型在接触处预测不准"从未通过诊断实验验证
2. **方案设计脱离文献**：TRD加权不符合MBRL文献（MOPO证明reward penalty更有效）
3. **MoE假设过强**：DreamerV3的Block GRU (deter=4096, 8 blocks)容量可能已经足够
4. **缺少诊断驱动**：应该先诊断瓶颈再设计方案，而非先设计方案再验证

### 后续方向（待定）

当前处于方向重新评估阶段，选项包括：
1. 对 DreamerV3 在操作任务上做系统诊断，确认真正瓶颈后再针对性改进
2. 参考 DreamerV4 (Transformer 架构) 或 TD-MPC2 (planning) 换更强底座
3. 保留 TRD 但改为 reward penalty（类 MOPO），去掉 MoE

需要读论文（DreamerV4, TD-MPC2）后再决定方向。
