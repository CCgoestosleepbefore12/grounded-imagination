# Grounded Imagination

## 项目概述
基于 DreamerV3 (JAX 官方版) 的 MBRL 框架，解决接触丰富操作任务中世界模型想象失真问题。
三个核心组件：MoE Dynamics + TRD (Transition Realism Discriminator) + DAgger 主动纠正。
详细方案见 docs/spec.md

## 目录结构
- dreamerv3/       — DreamerV3 官方 JAX 代码（需修改 agent.py, rssm.py, main.py）
- embodied/        — DreamerV3 依赖的通用 RL 工具库
- grounded/        — 我们新增的模块（MoE, TRD, DAgger, 优先级采样, 诊断工具）
- configs/         — 超参数配置
- scripts/         — 训练/评估脚本
- tests/           — 单元测试
- docs/            — spec.md (论文方案)

## 代码规范
- Python 3.10+, JAX + flax/ninjax
- 类型注解必须写
- 新模块必须有对应 test
- 所有张量操作注明 shape（注释里写 # (B, D) 这种）
- 伪代码是 PyTorch 风格（spec 中），实现用 JAX 风格

## 依赖
- 基础架构: DreamerV3 JAX 官方版 (danijar/dreamerv3)
- 环境: dm_control, metaworld, mani_skill2
- JAX 参考库: google-research/vmoe (MoE Router), GANs-JAX (Spectral Norm)

## 关键超参数 (详见 docs/spec.md 第五节)
- H_max = 20, tau = 0.15, S_warm = 5000
- train_ratio = 512, K_trd = 16, K_dagger = 128
- PER: alpha = 0.6, beta = 0.4 -> 1.0
