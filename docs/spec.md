# Grounded Imagination: Learning When and How to Trust World Model Rollouts for Contact-Rich Manipulation

> 面向接触丰富操作任务的可信想象框架：MoE 世界模型 + 转移真实性判别器 + DAgger 主动纠正

---

## 一、论文定位

### 标题方向

**"Grounded Imagination: Learning When and How to Trust World Model Rollouts for Contact-Rich Manipulation"**

### 一句话概括

> 世界模型的想象数据在简单操作中有效，但在接触丰富的复杂操作中严重失真导致策略训练失败。我们提出 Grounded Imagination 框架，包含三个互补组件：MoE 动力学让预测本身更准，Transition Realism Discriminator (TRD) 判断哪些想象可信，DAgger 主动纠正机制让世界模型持续修正不准的区域。以少量额外交互（DAgger 纠正，约增加 ~3% 环境步数）换取显著性能提升，让 DreamerV3 在之前完全失败的 pick-place 等接触任务上取得成功。

### 目标会议

- **首选**: NeurIPS 2026 / ICML 2027
- **备选**: CoRL 2026（对 manipulation + 真机更友好）
- **保底**: ICLR 2027

---

## 二、问题定义

### 2.1 出发点：已有实验证据

用 DreamerV3（8×A100, 1.1M steps, train_ratio=512, image 64×64）跑 8 个任务，结果清晰分两类：

| 类型 | 任务 | 最终分 | 最佳分 | 收敛 |
|------|------|--------|--------|------|
| 成功 | DMC cup_catch | 968 | 1000 | Yes |
| 成功 | DMC finger_spin | 987 | 995 | Yes |
| 成功 | DMC reacher_easy | 974 | 1000 | Yes |
| **失败** | **DMC manipulator_bring_ball** | **0** | **236** | **No** |
| 成功 | MetaWorld reach | 4885 | 4904 | Yes |
| 成功 | MetaWorld push | 4590 | 4756 | Yes |
| 成功 | MetaWorld door-open | 4584 | 4624 | Yes |
| **失败** | **MetaWorld pick-place** | **15** | **1062** | **No** |

失败任务的共同特征：需要多阶段接触转换（接近 → 接触 → 抓取 → 搬运）。

### 2.2 根本原因分析

```
世界模型想象链:
z₁ → ẑ₂ → ẑ₃ → ... → ẑ_k → ẑ_{k+1} → ... → ẑ_H
                        ↑
                    接触发生
                    动力学突变
                    预测误差骤增

接触前: 动力学近似线性（空中移动），误差小 → 想象可信
接触后: 动力学不连续（碰撞/摩擦/耦合），误差大 → 想象是"幻觉"

但 DreamerV3 对所有想象步一视同仁 → 策略被错误梯度污染 → 部署失败
```

### 2.3 形式化

- 真实 MDP: M = (S, A, P, R, γ)
- 世界模型: M̂ = (S, A, P̂, R̂, γ)
- 策略在真实环境: J(π), 在想象中: J̃(π)
- 单步模型误差: ε(s) = D_TV(P(·|s,a) ‖ P̂(·|s,a))

现有 MBRL 的性能 bound（MBPO 理论）:

```
|J(π) - J̃_k(π)| ≤ 2r_max [γ^k · ε_π + k/(1-γ) · (ε_m + ε_π)]
```

其中 ε_m = max_t E_π[ε(s_t)] 被最不准的状态（接触瞬间）主导。

**问题**: 即使 99% 的状态预测很准，bound 也被 1% 的接触状态搞崩。
**目标**: state-dependent 的更紧 bound + 让想象数据真正可用的方法。

---

## 三、方法：Grounded Imagination Framework

### 3.1 三个组件各解决一个问题

```
问题 1: 世界模型用一套动力学预测所有情况 → 在动力学变化大时不准
解决:   MoE Dynamics — 多个专家子模型，不同动力学自动分配不同专家

问题 2: 不知道哪些想象数据可信
解决:   TRD — 判别想象转移是否接近真实

问题 3: 世界模型在策略实际访问的区域训练不充分（分布偏移）
解决:   DAgger 主动纠正 — 在 TRD 标记的不可信区域主动收集真实数据
```

三者关系:

```
              MoE 世界模型
              (让预测更准 — 治本)
                  │
                  ▼ 产生想象数据
              TRD 判别器
              (判断哪里还不准 — 诊断)
               ╱        ╲
              ▼          ▼
     自适应想象          DAgger 纠正
   (不准的数据降权)    (不准的区域补数据 — 治本)
        │                   │
        ▼                   ▼
   策略训练更有效      世界模型越来越准
        │                   │
        └──────→ 正反馈循环 ←┘
```

### 3.2 组件一：MoE Dynamics World Model

#### 动机

操作任务中，不同时刻的动力学本质不同:
- 手在空中移动 → 简单运动学，近似线性
- 手推物体滑动 → 摩擦力主导，非线性
- 手抓住物体抬起 → 刚体约束，手物耦合
- 物体从手中滑落 → 自由下落 + 碰撞

一个 MLP 硬要学所有这些模式，能力必然打折。

#### 架构：替换 RSSM 中的 dynamics MLP

```
标准 RSSM:
  h_{t+1} = f(h_t, z_t, a_t)              ← 一个 MLP

MoE RSSM:
  h_{t+1} = Σ_k w_k(h_t, z_t, a_t) · f_k(h_t, z_t, a_t)

  其中:
    f_k = 第 k 个动力学专家 (MLP)
    w_k = softmax(Router(h_t, z_t, a_t))   ← Top-2 routing
```

#### 同构专家设计

```
默认: 4 个结构相同的 MLP, 不同随机初始化
  Expert 1: MLP(h+z+a → hidden → hidden → h_dim)  seed=1
  Expert 2: 同结构                                  seed=2
  Expert 3: 同结构                                  seed=3
  Expert 4: 同结构                                  seed=4

原理: 不同初始化 → 训练中对称性破缺 → 不同专家自动特化到不同动力学模式
     (与 Switch Transformer 等工作中观察到的现象一致)

专家数量: 默认 4, A10 实验测试 K=2,4,8 的最优值
```

#### 实现要点

```python
class MoEDynamics(nn.Module):
    def __init__(self, h_dim, z_dim, a_dim, num_experts=4):
        # 同构专家, 不同初始化
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim + z_dim + a_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, h_dim),
            ) for _ in range(num_experts)
        ])
        # Router
        self.router = nn.Sequential(
            nn.Linear(h_dim + z_dim + a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )
    
    def forward(self, h, z, a):
        x = torch.cat([h, z, a], dim=-1)
        logits = self.router(x)
        weights = F.softmax(logits, dim=-1)           # soft routing
        
        # Top-2 for efficiency
        top2_weights, top2_idx = weights.topk(2, dim=-1)
        top2_weights = top2_weights / top2_weights.sum(dim=-1, keepdim=True)
        
        # 加权组合
        output = sum(
            top2_weights[:, i:i+1] * self.experts[idx](h, z, a)
            for i, idx in enumerate(top2_idx.unbind(dim=-1))
        )
        return output
```

#### 辅助损失：Load Balancing

防止所有输入都被路由到同一个专家:
```
L_balance = CV(load)²    # load 是每个专家的使用频率，CV 是变异系数
```

#### 与已有工作的区别

| 已有工作 | 区别 |
|---------|------|
| Neural Hybrid Automata (NeurIPS 2021) | 硬离散切换 + Neural ODE，没用在操作任务上 |
| Jin & Posa (T-RO 2024) | 物理结构化 LCS，不是 learned latent world model |
| Switch Transformer / MoE LLM | 在 FFN 层用 MoE，不是在动力学预测层 |
| 标准 DreamerV3 | 单一动力学 MLP，没有专家分工 |

### 3.3 组件二：Transition Realism Discriminator (TRD)

#### 设计理念

回答: "这个转移 (z_t, a_t, z'_{t+1}) 看起来像不像真实环境中会发生的？"

#### 在潜在空间工作

| | 潜在空间 | 观测空间 |
|--|---------|---------|
| 维度 | ~200 维 | 64×64×3 = 12288 维 |
| 语义 | 直接包含动力学信息 | 大量与动力学无关的像素信息 |
| 计算 | 快，无需解码 | 慢 |

#### 实现

```python
class TransitionRealismDiscriminator(nn.Module):
    def __init__(self, z_dim, a_dim, hidden=256):
        self.net = nn.Sequential(
            SpectralNorm(nn.Linear(z_dim * 2 + a_dim, hidden)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Linear(hidden, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, z_t, a_t, z_next):
        x = torch.cat([z_t, a_t, z_next], dim=-1)
        return self.net(x)
```

#### 训练

```python
# 正样本: replay buffer 中的真实转移，encoder 编码到潜在空间
z_real, a_real, z_next_real = encode(replay_buffer.sample())

# 负样本: 世界模型从相同 (z, a) 预测下一状态
z_next_imagined = moe_world_model.predict(z_real, a_real)

# 标准二分类损失 + label smoothing
L_TRD = -E[0.9 · log D(z_real, a_real, z_next_real)]
        -E[log(1 - D(z_real, a_real, z_next_imagined))]
```

#### TRD 的潜在空间非平稳性

潜在空间 z 随世界模型更新在变化, TRD 必须跟上这个变化, 否则会失效。

解决: TRD 是**持续训练的** (每 K_trd=16 次模型更新训练 1 次 TRD),
与世界模型**同步演化**, 而非训一次就固定。

为什么持续训练够用:
- 世界模型单步更新引起的潜在空间漂移很小 (SGD 小步更新)
- TRD 每 16 步更新一次, 足以跟上这个速度
- TRD 结构简单 (3 层 MLP), 适应速度快
- 预热阶段不启用 TRD, 避免在潜在空间剧烈变化的早期引入噪声

#### 防止 TRD 过强（GAN 训练的经典问题）

如果 TRD 完美判别 → 所有想象数据得分=0 → 框架退化为纯 model-free

防止措施:
1. **Spectral Normalization**: 限制判别器 Lipschitz 常数
2. **Label Smoothing**: 真实标签用 0.9 而非 1.0
3. **训练比例控制**: 每训练世界模型 K_trd=16 步，TRD 只训练 1 步
4. **梯度惩罚（可选）**: WGAN-GP 风格

#### 自适应想象 Rollout

```python
# 标准 DreamerV3: 固定 H=15 步, 所有步等权重

# 我们: 自适应步数, 累积信任加权
z = initial_state
cumulative_trust = 1.0

for t in range(H_max):
    a = actor(z)
    z_next = moe_world_model.step(z, a)
    
    # TRD 评估这一步
    trust_t = TRD(z, a, z_next).detach()
    cumulative_trust *= trust_t
    
    weights[t] = cumulative_trust
    
    if cumulative_trust < τ:
        break
    
    z = z_next

# 加权策略训练
actor_loss = -weighted_mean(advantages, weights)
critic_loss = weighted_mean((V - targets)², weights)
```

#### 累积信任的直觉

```
想象步:    1     2     3     4     5     6     7     8
TRD 分:   0.95  0.93  0.90  0.88  0.35  0.40  0.60  0.70
                                    ↑
                                接触发生
                                预测突然变差

累积信任:  0.95  0.88  0.80  0.70  0.24  0.10  0.06  0.04
                                    ↑
                                接触后所有步都被大幅降权

效果: 接触前的想象数据高权重利用, 接触后的"幻觉"几乎不用
```

用累积乘积而非单步分数的原因: 一旦某步预测偏了, 后续所有步都建立在错误基础上, 即使后面某步 TRD 分数"恢复", 那也是在错误状态上的伪准确。

### 3.4 组件三：DAgger 主动纠正

#### 与经典 DAgger 的对应关系

| DAgger（经典） | Grounded Imagination |
|---|---|
| Learner = 策略 π | Learner = 世界模型 M̂ |
| Expert = 真实专家 | Expert = 真实环境动力学 |
| 分布偏移 = π 访问的状态不在专家演示中 | 分布偏移 = π 访问的状态不在 M̂ 训练分布中 |
| 纠正 = 去专家那里要标签 | 纠正 = 去环境中执行，拿真实转移 |
| 数据聚合 = 新标签加入训练集 | 数据聚合 = 纠正数据加入 replay buffer |

#### 核心机制：TRD 引导的主动数据收集

**关键问题**: 想象在潜在空间 z_0 → z_1 → ... → z_k 发生, TRD 在 z_k 处标记"不可信",
但 env.set_state() 需要 (qpos, qvel)。z_k 是纯想象的, 没有对应的 MuJoCo 状态。

**解决: 动作回放法 + 累积信任保证**

想象总是从 replay buffer 采样的真实状态 z_0 开始, z_0 对应的 (qpos_0, qvel_0) 已知。
自适应 rollout 的累积信任截断保证: 能走到 z_k 才被标记不可信 → 前面 0~k-1 步都可信
→ 真实环境回放同样动作到达的 s_k ≈ 想象的 z_k。

```python
# DAgger 纠正循环

# 1. 策略在世界模型中想象（自适应 rollout, 记录起始真实状态和动作序列）
z_0, (qpos_0, qvel_0) = sample_initial_state(replay_buffer)
imagined_traj = adaptive_rollout(moe_world_model, policy, TRD, z_0, H_max, tau)
# imagined_traj 包含: z_序列, a_序列, trust_序列, 截断点 k

# 2. 找到累积信任刚跌破阈值的 transitions（想象刚开始不准的边界）
critical_transitions = find_trust_boundary(imagined_traj, tau)

# 3. 动作回放: 从真实起点出发, 用相同动作序列到达关键状态
for (k, a_k) in critical_transitions:
    env.set_state(qpos_0, qvel_0)             # 回到真实起点
    for i in range(k):
        s, r = env.step(imagined_traj.actions[i])  # 回放前 k 步动作
    # 现在到达真实的 s_k (≈ 想象的 z_k, 因为前 k 步可信)
    s_k = env.get_state()
    
    # 收集关键转移的正确答案: 策略选了 a_k, 真实 s_{k+1} 是什么?
    s_next_real, r_real = env.step(a_k)
    
    # 高优先级加入 replay buffer
    replay_buffer.add(s_k, a_k, r_real, s_next_real, priority=1.0)

# 4. 用纠正数据重点更新世界模型
moe_world_model.train(replay_buffer, prioritized=True)
```

**为什么这是正确的 DAgger**:
- DAgger 的核心: 在 learner (世界模型) 当前犯错的 (s, a) 上, 向 expert (真实环境) 要正确答案
- 不是探索新动作, 而是让世界模型把策略当前会走的路预测准
- 策略选了 a_k → 世界模型预测错了 → 去真实环境拿 (s_k, a_k) 的正确 s_{k+1}

**累积信任与动作回放的配套关系**:
- 累积信任保证: 走到 z_k 才不可信 → 前面都准 → 回放路径有效
- 只在"刚开始不准"的边界处纠正 → 最有价值的纠正位置
- 如果中途就不准, 累积信任早已跌破 τ, rollout 已截断, 不会到达 z_k

#### 为什么不违背仿真效率原则

```
标准 DreamerV3:  每步都和环境交互, 不管有没有必要
                 有些交互在模型已经很准的区域, 浪费了

DAgger-guided:   只在 TRD 标记的"不可信区域"去环境交互
                 在已经准的区域, 完全用想象

以少量额外交互 (~3%) 换取显著提升, 且这些交互精准投放在世界模型最需要的区域
```

#### 仿真 vs 真机的两种模式

```
仿真模式（完整 DAgger）:
  - 可以 env.set_state() 瞬移到任意状态
  - 直接在不可信状态收集真实数据
  - 数据纠正最高效

真机模式（Soft DAgger）:
  - 不能瞬移, 但可以引导探索策略
  - 探索策略偏向不可信区域（加 exploration bonus）
  - exploration_bonus(s) ∝ 1 / TRD_score(s)
  - 效果弱于完整 DAgger, 但不需要 set_state
```

---

## 四、整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                   Grounded Imagination Framework                 │
│                                                                 │
│                        观测 o_t + 动作 a_t                       │
│                              │                                  │
│                              ▼                                  │
│                     ┌── RSSM Encoder ──┐                        │
│                     │   h_t, z_t        │                        │
│                     └──────────────────┘                        │
│                              │                                  │
│                ┌─────────────┼──────────────┐                   │
│                ▼             ▼              ▼                   │
│           ┌─────────┐  ┌─────────┐  ┌──────────┐              │
│           │Expert 1  │  │Expert 2  │  │Expert N   │              │
│           │(同构 MLP)│  │(同构 MLP)│  │(同构 MLP) │              │
│           │seed=1    │  │seed=2    │  │seed=N     │              │
│           └────┬────┘  └────┬────┘  └────┬─────┘              │
│                │            │            │                      │
│                └────────┬───┴────────────┘                      │
│                    Router│(Top-2 Soft Routing)                   │
│                         ▼                                       │
│                   ẑ_{t+1} (MoE 预测)                            │
│                         │                                       │
│              ┌──────────┴──────────┐                            │
│              ▼                     ▼                            │
│     ┌──────────────┐      ┌──────────────┐                     │
│     │     TRD      │      │  Actor π(z)  │                     │
│     │ D(z,a,ẑ')→c_t│      │  Critic V(z) │                     │
│     └──────┬───────┘      └──────────────┘                     │
│            │                      ▲                             │
│     ┌──────┴──────┐               │                             │
│     ▼             ▼               │                             │
│  c_t > τ?      c_t < τ?          │                             │
│     │             │               │                             │
│     ▼             ▼               │                             │
│  继续想象    ┌──────────┐         │                             │
│  权重=c_t   │ DAgger    │         │                             │
│     │       │ 纠正      │         │                             │
│     │       │ env.set() │         │                             │
│     │       │ 收集真实   │         │                             │
│     │       │ 转移      │         │                             │
│     │       └────┬─────┘         │                             │
│     │            ▼               │                             │
│     │    Replay Buffer           │                             │
│     │    (优先级采样)             │                             │
│     │            │               │                             │
│     │            ▼               │                             │
│     │    更新 MoE 世界模型 ──────→│                             │
│     │                            │                             │
│     └──── 加权策略训练 ──────────→┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、完整算法

```
算法: Grounded Imagination

输入: 环境 env, 最大步数 N=1.1M
超参数:
  H_max = 20                 // 最大想象步数 (DreamerV3 默认 H=15, 我们放宽到 20,
                              //   因为自适应截断会在不可信时提前停止, 可信区域可以走更远)
  τ = 0.15                   // 累积信任截断阈值
                              //   太高 (如 0.5): 截断太早, 可用想象数据太少
                              //   太低 (如 0.01): 截断太晚, 允许不可信想象参与训练
                              //   0.15 ≈ 连续 ~10 步 TRD=0.85 后截断, 或 1 步 TRD=0.15 立即截断
  S_warm = 5000              // 预热步数: 纯随机探索, 只训世界模型+策略, 不启用 TRD/DAgger
                              //   需要足够数据让潜在空间初步稳定, 再启用 TRD
  train_ratio = 512          // 每 1 步环境交互做 512 次模型更新 (与 DreamerV3 一致)
  C = 1                      // 每 C 步环境交互触发一轮更新
  K_trd = 16                 // TRD 训练频率: 每 16 次模型更新训练 1 次 TRD
                              //   → 每轮 512/16 = 32 次 TRD 更新
                              //   → 世界模型 : TRD = 16:1, 防止判别器过强
  K_dagger = 128             // DAgger 纠正频率: 每 128 次模型更新纠正 1 次
                              //   → 每轮 512/128 = 4 次 DAgger 纠正
                              //   → 每次回放 ~3-8 步, 额外环境交互 ~32 步/轮
输出: 策略 π_φ

初始化:
  MoE 世界模型 M_θ (RSSM encoder + MoE dynamics + decoder + reward)
  策略 π_φ (actor)
  价值函数 V_ψ (critic)
  TRD 判别器 D_ω
  Replay buffer B ← ∅
  // 注: B 的每条记录存储 (o_t, a_t, r_t, o_{t+1}, qpos_t, qvel_t)
  // 相比标准 DreamerV3, 额外存 MuJoCo 状态 (qpos, qvel), 供 DAgger 动作回放使用
  // 额外存储开销: qpos (~30维) + qvel (~30维) << 图片观测 (64×64×3), 可忽略

════════ 预热阶段（前 S_warm 步）════════
for step = 1 to S_warm:
    a_t = random_action()                  // 随机探索
    qpos_t, qvel_t = env.get_state()       // 记录 MuJoCo 状态
    o_{t+1}, r_t = env.step(a_t)
    B.add(o_t, a_t, r_t, o_{t+1}, qpos_t, qvel_t)
    标准训练: MoE 世界模型 + 策略 + 价值函数（无 TRD, 无 DAgger）

════════ 主训练阶段 ════════
for step = S_warm to N:

    // ──── 环境交互 ────
    a_t = π_φ(o_t) + exploration_noise
    qpos_t, qvel_t = env.get_state()       // 记录 MuJoCo 状态
    o_{t+1}, r_t = env.step(a_t)
    B.add(o_t, a_t, r_t, o_{t+1}, qpos_t, qvel_t)

    // ──── 每 C 步做一轮完整更新 ────
    if step % C == 0:
      for update_step = 1 to train_ratio:

        // (1) MoE 世界模型训练（Proportional PER）
        //
        // 优先级计算:
        //   普通数据:   p_i = 1 - TRD_score(s, a, s'_pred)  (TRD 分越低 → 越不准 → 优先级越高)
        //   DAgger 数据: p_i = p_max = 1.0                   (纠正数据最高优先)
        //
        // 采样概率:  P(i) = p_i^α / Σ p_j^α,  α = 0.6
        //   α=0 均匀采样, α=1 完全按优先级, α=0.6 适度偏向不准区域
        //
        // 重要性采样权重 (修正采样偏差):
        //   w_i = (N · P(i))^(-β) / max_j(w_j)
        //   β 从 0.4 线性退火到 1.0 (早期快速学习, 后期无偏收敛)
        //
        batch, is_weights = B.prioritized_sample(batch_size, α=0.6, β=anneal(0.4→1.0))
        L_model = weighted_mean(ELBO_loss(M_θ, batch), is_weights) + λ_balance · L_balance
        更新 M_θ

        // (2) TRD 训练（控制频率）
        if update_step % K_trd == 0:
            z_real, a, z_next_real = encode(batch)
            z_next_pred = M_θ.predict(z_real, a)
            L_TRD = BCE_smoothed(D_ω, real=(z_real, a, z_next_real),
                                       fake=(z_real, a, z_next_pred))
            更新 D_ω

        // (3) 自适应想象 + 加权策略训练
        z_0, (qpos_0, qvel_0) = 从 B 采样初始状态并编码（同时记录真实 MuJoCo 状态）
        traj, weights = adaptive_rollout(M_θ, π_φ, D_ω, z_0, H_max, τ)
        L_actor  = -weighted_mean(advantages(traj), weights)
        L_critic = weighted_mean((V(traj) - targets)², weights)
        更新 π_φ, V_ψ

        // (4) DAgger 主动纠正（动作回放法）
        if update_step % K_dagger == 0:
            // 找到累积信任刚跌破阈值的边界转移
            critical = find_trust_boundary(traj, weights, τ)
            for (k, a_k) in critical:
                // 从真实起点回放动作序列到达 s_k
                env.set_state(qpos_0, qvel_0)
                for i in 0 to k-1:
                    env.step(traj.actions[i])
                // 收集关键转移的正确答案
                s_k = env.get_state()
                s_next_real, r = env.step(a_k)
                B.add(s_k, a_k, r, s_next_real, priority=1.0)

    // ──── 日志 ────
    记录: 平均 TRD score, 平均有效想象步数, episode return,
          MoE router 分布, 各 expert 使用率

return π_φ


════════ 子程序: adaptive_rollout ════════
function adaptive_rollout(M, π, D, z_0, H_max, τ):
    z = z_0
    cumulative_trust = 1.0
    trajectory = []
    weights = []
    
    for t = 1 to H_max:
        a = π(z)
        z_next = M.step(z, a)                    // MoE dynamics
        r = M.reward(z_next)
        v = V(z_next)
        
        trust_t = D(z, a, z_next).detach()        // TRD 评分, 不回传梯度
        cumulative_trust = cumulative_trust * trust_t
        
        trajectory.append((z, a, r, v, z_next))
        weights.append(cumulative_trust)
        
        if cumulative_trust < τ:
            break
        z = z_next
    
    return trajectory, weights
```

---

## 六、理论分析

### 6.1 定理 1: 自适应想象的性能 bound

**Setup:**
- 单步模型误差在状态 s 下为 ε(s) = D_TV(P(·|s,a) ‖ P̂(·|s,a))
- TRD 截断阈值 τ, TRD 分类误差 ε_D
- 自适应截断点 h*(s) = max{h : cumulative_trust ≥ τ}

**定理 1:**
```
|J(π) - J̃_adaptive(π)| ≤ 2r_max · E_{s~d_π} [
    γ^{h*(s)} / (1-γ)                     ... 截断后残余误差
    + h*(s) · (τ + ε_D)                    ... 截断前控制误差
]
```

**对比标准 bound:**
```
标准 (固定 H 步):  |J - J̃| ≤ O(H · max_s ε(s))    ← 被最差状态主导
自适应 (我们的):    |J - J̃| ≤ O(E_s[h*(s)] · τ)     ← 按状态自适应
```

当 ε(s) 在不同状态差异大时（操作任务典型情况），我们的 bound 严格更紧。

**直觉:**
```
自由空间:   ε(s) ≈ 0.01, h*(s) = 15+ （可以长步想象）
接近物体:   ε(s) ≈ 0.05, h*(s) = 8   （中等步想象）
接触瞬间:   ε(s) ≈ 0.50, h*(s) = 2   （极短想象）
抓握搬运:   ε(s) ≈ 0.20, h*(s) = 5   （短步想象）

标准 bound 被 ε=0.50 主导, 必须全局短 rollout
我们的 bound 在每个区域用最优步数
```

### 6.2 定理 2: DAgger 纠正的收敛性

若世界模型类 M 的 capacity 足以表示真实动力学, 在 DAgger 纠正下:

```
E_{d_π}[ε(s)] ≤ ε_approx + O(√(D_KL(d_π ‖ d_buffer) / N_corrected))
```

其中 N_corrected 是 DAgger 纠正的有效样本数。

与均匀采样相比, 当 d_π ≠ d_buffer（策略分布和 buffer 分布不匹配）时, DAgger 纠正用更少的样本达到同等精度。

### 6.3 推论: 自我改善性质

```
正反馈循环:
TRD 检测不准区域 → DAgger 收集纠正数据 → MoE 世界模型更新
→ 预测更准 → TRD 分数提高 → 可信想象范围扩大
→ 策略获得更多有效数据 → 策略改善 → 访问新区域 → 继续改善

收敛: 当 TRD 无法区分想象和真实（score ≈ 0.5 everywhere）
→ 框架退化为标准 DreamerV3 + MoE
→ 这恰好是世界模型已经足够准的理想状态
```

---

## 七、实验设计

### 7.1 环境与任务

```
第一梯队（已有 baseline）:
  DMC:       cup_catch, finger_spin, reacher_easy, manipulator_bring_ball
  MetaWorld: reach, push, door-open, pick-place

第二梯队（补充难度和多样性）:
  MetaWorld: box-close, shelf-place, bin-picking
  ManiSkill2: pick-cube, stack-cube, peg-insertion

选择逻辑:
  - 第一梯队: 有现成 baseline 数据, 直接对比
  - 第二梯队: 证明方法泛化性
  - 包含从简单到困难的完整梯度
```

### 7.2 Baselines

```
Model-Based:
  1. DreamerV3                  — 已有, 主要对比对象
  2. DreamerV3 + Ensemble       — 证明不是加 ensemble 就行
  3. TD-MPC2                    — 另一类世界模型方法
  4. MBPO                       — 经典自适应 MBRL
  5. BIRD                       — 互信息桥接想象与现实, 置信度加权策略优化

Model-Free:
  6. SAC                        — model-free 参考线

消融对比:
  7. DreamerV3 + MOPO-style     — 奖励惩罚 vs 我们的 TRD 加权
  8. DreamerV3 + MOReL-style    — 二值截断 vs 我们的 soft 截断
  9. Ours (完整 Grounded Imagination)
```

### 7.3 核心实验

#### 实验 1: 主结果（正常训练对比）

所有方法跑 1.1M steps（和已有实验一致）, 5 seeds, 报告均值 ± 标准差。

**期望**: 简单任务方法间差异不大; 复杂任务（pick-place, bring_ball）我们大幅领先。

**核心数字**: DreamerV3 pick-place 15 分 → 我们的方法 X 分（X 需要显著高于 15）。

#### 实验 2: 样本效率（限制真实交互预算）

限制真实环境交互为 10k / 50k / 100k / 500k 步。画**交互步数 vs 成功率**曲线。

**期望**: 同样预算下, 我们的方法成功率显著更高。

**意义**: 直接对应真机场景——真机上只能做有限次交互。

#### 实验 3: Ablation Study

| 编号 | 配置 | 验证目标 |
|------|------|---------|
| A1 | 完整方法 (MoE + TRD + DAgger) | Full system |
| A2 | 去掉 MoE, 用标准 RSSM | MoE 的独立贡献 |
| A3 | 去掉 TRD, 用 ensemble disagreement | TRD vs ensemble |
| A4 | 去掉自适应截断, 固定 H=15 | 截断的贡献 |
| A5 | 去掉加权, 所有步等权 | 加权的贡献 |
| A6 | 去掉 DAgger 纠正 | DAgger 的贡献 |
| A7 | 去掉优先级模型训练, 均匀采样 | 优先采样的贡献 |
| A8 | MOPO 风格奖励惩罚替代 TRD | 我们的方法 vs MOPO 思路 |
| A9 | MOReL 风格二值截断替代 soft 截断 | soft vs hard |
| A10 | MoE expert 数量: K=2,4,8 | 最优 expert 数量 |
| A11 | 异构 MoE (不同深度/宽度的专家) vs 同构 MoE | 异构设计是否优于同构 |
| A12 | BIRD-style 置信度加权（用世界模型预测似然替代 TRD 做权重） | TRD vs model likelihood 作为置信度度量 |

**Ablation 任务选择 (控制计算量):**
- 主结果 (实验 1): 全部 14 个任务 × 5 seeds = 70 runs
- Ablation (A1-A12): 只在 3 个代表性任务上跑 × 5 seeds = 180 runs
  - pick-place (困难, 接触丰富, 核心验证任务)
  - cup_catch (简单, 确保不退化)
  - push (中等, 有接触但不需要抓取)
- 总计: 70 + 180 = 250 runs, 约 2000 GPU 小时 (8×A100 约 10 天)

**关键对比:**
- A1 vs A2 → MoE 是否必要
- A1 vs A3 → TRD 是否优于 ensemble
- A1 vs A6 → DAgger 的增量贡献
- A1 vs A2 vs A6 vs (A2+A6) → 三组件是否正交叠加
- A1 vs A12 → TRD (learned discriminator) 是否优于 BIRD-style model likelihood 作为置信度度量

#### 实验 4: 诊断分析（论文核心可视化）

**图 1 (Motivation, 放 Introduction):**
DreamerV3 在 pick-place 上世界模型预测误差 vs rollout 步数。标注接触时间步。展示误差在接触瞬间突变。

**图 2 (Hero figure, 放 Method/Results):**
TRD 分数随时间变化 + 有效想象步数变化。展示框架自动在接触前长步想象、接触后短步想象。

**图 3 (MoE 分析):**
Router weights 随时间的变化。观察同构专家是否自动分化到不同动力学模式。
如果 router 自动学会在不同阶段切换专家, 这就是论文的 highlight — "我们从未告诉它什么是接触, 但它自己学会了分工"。

```
时间轴:  ──────────────────────────────────────→
Expert 1:  ████████░░░░░░░░░████████░░░░░░  (可能对应空中运动)
Expert 2:  ░░░░░░░░████░░░░░░░░░░░░░░░░░░  (可能对应碰撞)
Expert 3:  ░░░░░░░░░░░░████████░░░░░░█████  (可能对应抓握)
                        ↑                 ↑
                    接触发生           再次抓取
(注: 专家的具体分工由训练自动决定, 以上为预期的理想结果)
```

**图 4 (收敛分析):**
随训练进行: 平均 TRD 分数逐渐提高 + 平均有效想象步数增长 → 证明"自我改善"。

**图 5 (DAgger 效果):**
世界模型在接触区域的预测误差: 有 DAgger vs 无 DAgger, 随训练迭代的变化。

#### 实验 5: Scaling 实验

固定 pick-place 任务, 网络参数从 0.5M → 2M → 5M → 10M:
- DreamerV3 是否 scaling 后仍然失败?
- 我们的方法是否 scaling 后持续提升?

#### 实验 6: 真机部署

```
方案 A（推荐, 如果有机械臂）:
  - 选 1-2 个任务（如 pick-place, peg-insertion）
  - 少量真机交互（50-100 episodes）+ 大量想象训练
  - 真机模式: Soft DAgger（不能 set_state, 用 exploration bonus 引导）
  - 对比: 标准 DreamerV3 vs 我们的方法
  - 视频展示真机成功执行

方案 B（如果没有机械臂）:
  - 仿真中模拟 sim-to-real gap（观测噪声 + 动力学随机化）
  - 证明方法在 domain gap 下的鲁棒性
```

### 7.4 评估指标

```
主指标:
  - Episode return (累积奖励)
  - Success rate (任务成功率, MetaWorld/ManiSkill 提供)

分析指标:
  - 平均 TRD score (想象质量随训练变化)
  - 有效想象步数 (自适应截断后的平均 rollout 长度)
  - 世界模型预测误差 (1/5/10/20 步 MSE)
  - MoE router entropy (专家使用分布)
  - DAgger 纠正频率 (哪些区域被纠正最多)
  - Wall-clock time (确保额外开销可接受)

计算开销预估 (相对于原版 DreamerV3):
  ┌──────────────────┬─────────────────────────┬──────────────┐
  │ 组件             │ 额外计算                 │ wall-clock   │
  ├──────────────────┼─────────────────────────┼──────────────┤
  │ MoE Dynamics     │ dynamics 层 ~2.3x        │ ~13%         │
  │                  │ (Router + Top-2 Expert)  │              │
  │                  │ dynamics 占 RSSM ~20%    │              │
  ├──────────────────┼─────────────────────────┼──────────────┤
  │ TRD 推理         │ 每想象步 +1 次小 MLP     │ ~3%          │
  ├──────────────────┼─────────────────────────┼──────────────┤
  │ TRD 训练         │ 每 16 次模型更新训 1 次  │ ~1%          │
  ├──────────────────┼─────────────────────────┼──────────────┤
  │ DAgger 环境交互  │ ~32 步 MuJoCo/轮        │ <0.1%        │
  │                  │ (32×0.1ms vs 512×10ms)  │              │
  ├──────────────────┼─────────────────────────┼──────────────┤
  │ 总计             │                         │ ~17%         │
  └──────────────────┴─────────────────────────┴──────────────┘
  结论: 额外开销 ~17%, 可接受。需在实验中实测并报告。
```

---

## 八、代码实现路径

### 8.1 基于现有代码

```
代码基础:
  JAX 官方版 DreamerV3: https://github.com/danijar/dreamerv3
  语言: JAX + flax/ninjax (作者自研的 nn 抽象)
  选择理由:
    - 官方实现, baseline 数字可直接对齐
    - JAX JIT + XLA 训练速度快 ~1.5-2x, 大量实验 (5 seeds × 多 ablation) 受益显著
    - 8×A100 多卡利用率更高

已有代码:
  world_model/
  ├── AIL_from_visual_obs/    ← V-MAIL (不用改)
  └── dreamerv3/              ← DreamerV3 官方 JAX 版 (主要改这个)

需要修改的文件 (DreamerV3 JAX 代码库, 已确认):
  agent.py         ← 修改想象 rollout 逻辑, 加 TRD 评估 + 自适应截断
  rssm.py          ← RSSM dynamics MLP 替换为 MoE dynamics
  main.py          ← 加 TRD 训练循环, DAgger 纠正循环, 优先级采样

需要新增的文件 (JAX/flax 风格):
  moe_dynamics.py  ← MoE 动力学模块 (Router + 同构 Experts)
  trd.py           ← Transition Realism Discriminator
  dagger.py        ← DAgger 主动纠正逻辑 (动作回放法)
  prioritized.py   ← 优先级采样工具
  diagnostics.py   ← 诊断和可视化工具 (TRD 分析, router 分析, 误差曲线)
```

### 8.2 JAX 参考实现库

```
MoE 参考 (参考 Router + Top-k 逻辑, 不直接引入):
  - google-research/vmoe        — Google 官方 Vision MoE, Router 实现权威
    https://github.com/google-research/vmoe
  - jax-dropless-moe            — 轻量级 block-sparse MoE
    https://github.com/davisyoshida/jax-dropless-moe

GAN/判别器参考 (参考 Spectral Norm + 训练流程, 不直接引入):
  - GANs-JAX                    — Flax 实现多种 GAN, 含 Spectral Normalization
    https://github.com/lweitkamp/GANs-JAX
  - Equinox DCGAN example       — GAN 训练流程参考
    https://docs.kidger.site/equinox/examples/deep_convolutional_gan/

注: MoE 和 TRD 结构简单 (MoE ~80 行, TRD ~50 行), 自己写更干净。
    Flax 无内置 Spectral Normalization, 需从 GANs-JAX 参考实现。
```

### 8.3 实现优先级

```
第 1 周:  MoE dynamics 实现 + 集成到 RSSM
第 2 周:  TRD 模块 + 自适应 rollout + 加权训练
第 3 周:  DAgger 纠正逻辑 + 优先级采样
第 4 周:  在 cup_catch (简单) + pick-place (困难) 上验证全部组件
```

---

## 九、时间线

```
第 1 个月: 核心实现 + 初步验证
├── Week 1-2: MoE dynamics + TRD + 自适应想象 + 集成到 DreamerV3
├── Week 3:   DAgger 纠正 + 在 8 个已有任务上跑完整实验
└── Week 4:   初步结果分析, 确认方向有效
               ★ 关键决策点: pick-place 是否有显著提升?
               如果有 → 继续
               如果没有 → 诊断哪个组件不 work, 调整设计

第 2 个月: 完整实验 + 理论
├── Week 5-6: 补充 baseline (TD-MPC2, MBPO, SAC)
│             补充第二梯队任务
│             Ablation 实验 (A1-A12, 在 pick-place/cup_catch/push 上)
├── Week 7:   样本效率实验 (限制交互预算)
│             Scaling 实验
└── Week 8:   理论分析 (bound 推导和证明)
               诊断实验 + 全部可视化

第 3 个月: 真机 + 论文
├── Week 9-10: 真机部署 (选 1-2 个任务)
├── Week 11:   论文初稿
└── Week 12:   修改完善

第 4 个月 (缓冲): 应对延期 + 最终打磨 + 投稿
```

---

## 十、论文结构

```
1. Introduction (1.5 页)
   - 世界模型对 MBRL 的价值
   - 核心问题: 想象数据在复杂操作中失真
   - 图 1: DreamerV3 预测误差在接触瞬间突变的诊断图
   - 方案概述 + 贡献列表

2. Related Work (1 页)
   - Model-Based RL: MBPO, Dreamer, TD-MPC
   - 想象与现实的差距: BIRD (互信息), MOPO, MOReL, COMBO, RAMBO
   - 探索与主动学习: Plan2Explore, MAX, VIME
   - DAgger 与分布偏移: Ross et al., Venkatraman et al.
   - MoE in RL: 相关但不在世界模型动力学层

3. Preliminaries (0.5 页)
   - MBRL 形式化, DreamerV3/RSSM 简介
   - 现有 performance bound (MBPO)

4. Method: Grounded Imagination (3 页)
   4.1 MoE Dynamics World Model
   4.2 Transition Realism Discriminator
   4.3 Cumulative Trust & Adaptive Imagination
   4.4 DAgger-Guided Active Correction
   4.5 Complete Algorithm

5. Theoretical Analysis (1.5 页)
   5.1 自适应想象的 state-dependent 性能 bound
   5.2 DAgger 纠正的收敛分析
   5.3 与 DAgger reduction 的形式化联系

6. Experiments (3 页)
   6.1 Setup
   6.2 Main Results (表格 + 学习曲线)
   6.3 Sample Efficiency
   6.4 Ablation Study
   6.5 Diagnostic Analysis (TRD 可视化, MoE router 分析, 误差分析)
   6.6 Real Robot Experiments

7. Discussion & Limitations & Future Work (0.5 页)
   - TRD 训练稳定性
   - 计算开销 (MoE + TRD 的额外成本)
   - DAgger 在真机上的局限 (不能 set_state)
   - 对非操作任务的适用性
   - Future Work: 结构化潜在空间 (encoder 输出 z_hand + z_object) + 异构输入 MoE,
     让不同专家接收不同的物体级表征, 进一步提升耦合动力学建模能力

8. Conclusion
```

---

## 十一、贡献总结

| 层次 | 贡献 | 独创性 |
|------|------|--------|
| **架构** | MoE Dynamics — 同构专家动力学模型, 通过 Router + Load Balancing 让不同专家自动特化到不同动力学模式 | 在世界模型动力学层使用 MoE 是新的 |
| **度量** | TRD — 学习的转移真实性判别器, 替代启发式 ensemble disagreement, 提供连续可信度评分 | 用 GAN 判别器度量想象质量是新的角度 |
| **训练范式** | DAgger 主动纠正 — TRD 引导的有针对性真实数据收集, 解决世界模型的分布偏移 | 在现代 Dreamer 架构上实现真正的 DAgger 循环是新的 |
| **理论** | State-dependent 自适应想象性能 bound, 严格优于固定步数 bound; DAgger 收敛分析 | 现有 bound (MBPO/MOPO) 是 state-independent 的 |
| **实验** | 系统性诊断世界模型在接触操作中的想象失真 + 解决方案 + 真机部署 | 之前无系统分析 |

---

## 十二、风险与应对

| 风险 | 可能性 | 应对策略 |
|------|--------|---------|
| TRD 训练不稳定 | 中 | SpectralNorm + label smoothing + 控制训练比例; 最坏退回 ensemble disagreement |
| MoE router 退化（只用 1 个 expert）| 中 | Load balancing loss; 监控 router entropy |
| pick-place 提升不显著 | 中 | 逐组件诊断; 如果是探索问题加 Plan2Explore bonus |
| DAgger set_state 在某些环境不可用 | 低 | 退回 Soft DAgger (exploration bonus 引导) |
| 审稿人认为"三组件拼接" | 中-高 | 理论分析深度 + ablation 证明正交 + 诊断可视化 |
| 理论 bound 推导困难 | 低-中 | 从简化假设开始（线性世界模型）再扩展 |
| 真机部署出问题 | 中 | 预留第 4 个月缓冲; 最坏用 sim-to-real gap 仿真实验替代 |
| 计算开销被质疑 | 低 | 预估总开销 ~17% (MoE ~13%, TRD ~4%, DAgger <0.1%); 实验中报告 wall-clock time |

---

## 十三、核心参考文献

| 论文 | 会议 | 与本方案的关系 |
|------|------|--------------|
| DreamerV3 (Hafner et al.) | JMLR 2023 | 基础架构, 主要对比对象 |
| MBPO (Janner et al.) | NeurIPS 2019 | 分支 rollout + 理论 bound, 我们改进其 bound |
| MOPO (Yu et al.) | NeurIPS 2020 | 奖励惩罚 vs 我们的 TRD 加权, 理论对比 |
| MOReL (Kidambi et al.) | NeurIPS 2020 | 二值截断 vs 我们的 soft 截断 |
| COMBO (Yu et al.) | ICLR 2022 | CQL 风格保守 vs 我们的判别器引导 |
| RAMBO (Rigter et al.) | NeurIPS 2022 | 对抗鲁棒 vs 我们的主动纠正 |
| TD-MPC2 (Hansen et al.) | ICLR 2024 | 另一类 MBRL baseline |
| DAgger (Ross et al.) | AISTATS 2011 | 分布偏移理论基础 |
| Venkatraman et al. | ICML 2015 | DAgger 用于世界模型学习的先驱 |
| Plan2Explore (Sekar et al.) | ICML 2020 | 不确定性引导探索 |
| Neural Hybrid Automata | NeurIPS 2021 | 混合动力学先驱, 需讨论区别 |
| Switch Transformer | JMLR 2022 | MoE 架构参考 |
| BIRD (Zhu et al.) | NeurIPS 2020 | 互信息桥接想象与现实, 用 model likelihood 做置信度加权; 我们用 learned TRD 替代, 且额外改善世界模型本身 (MoE + DAgger) |
| Talvitie | AAAI 2014 | Hallucinated replay, 世界模型 on-policy 纠正 |

---

## 十四、一句话 Pitch

> DreamerV3 在 pick-place 上只得 15 分, 因为它的想象在接触瞬间变成了幻觉但仍被当作训练数据。我们教会世界模型三件事: 用不同的专家预测不同的物理 (MoE), 知道自己什么时候在胡说 (TRD), 以及主动去自己最不准的地方补课 (DAgger)。结果: pick-place 从 15 分到 X 分, 并成功部署到真实机器人。
