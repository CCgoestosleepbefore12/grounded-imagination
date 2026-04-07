# Grounded Imagination

**Learning When and How to Trust World Model Rollouts for Contact-Rich Manipulation**

> MoE Dynamics + Transition Realism Discriminator + DAgger Active Correction

Built on [DreamerV3](https://github.com/danijar/dreamerv3) (JAX).

---

## Problem

World model imagination works well for simple tasks but fails catastrophically on contact-rich manipulation (e.g., pick-place, bring-ball). The dynamics change abruptly at contact, causing prediction errors that compound over imagination rollouts and corrupt policy training.

## Approach

Three complementary components:

| Component | Problem Solved | Implementation |
|-----------|---------------|----------------|
| **MoE Dynamics** | Single MLP can't model multiple dynamics modes | Top-2 routed homogeneous Block GRU experts |
| **TRD** | Don't know which imagination steps are trustworthy | Spectral-normed discriminator scoring (z_t, a_t, z_next) |
| **DAgger Correction** | World model undertrained in policy-visited regions | Active data collection at low-trust states |

Additionally:
- **Discounted cumulative trust** (gamma=0.5) replaces naive cumprod to prevent over-penalization on easy tasks
- **TRD-based prioritized replay** focuses model training on inaccurate regions

## Project Structure

```
dreamerv3/              # DreamerV3 (minimal modifications)
  agent.py              #   +TRD init, +2 grounded function calls in loss()
  rssm.py               #   +MoE option in _core(), +balance loss
  configs.yaml          #   +grounded config block, +metaworld preset
  main.py               #   +metaworld env registration

grounded/               # Our contribution (all new code)
  moe_dynamics.py       #   MoE Block GRU experts + Top-2 router
  trd.py                #   Spectral-normed transition discriminator
  integration.py        #   TRD loss + trust weight computation helpers
  dagger.py             #   Trust boundary detection + action replay utilities
  prioritized.py        #   Priority computation + IS weights + beta schedule
  diagnostics.py        #   Analysis functions for paper visualizations

embodied/
  envs/metaworld.py     #   MetaWorld wrapper (proprio + optional image)
  run/train.py          #   +DAgger correction loop, +warmup

tests/                  # 44 unit tests
scripts/                # Multi-GPU training launch scripts
docs/
  spec.md               # Full research specification
  experiment_log.md     # Experiment results and findings
```

## Quick Start

```bash
# Install
git clone https://github.com/CCgoestosleepbefore12/grounded-imagination.git
cd grounded-imagination
pip install -r requirements.txt
pip install -e .

# Run with Grounded Imagination enabled
MUJOCO_GL=egl python -m dreamerv3.main \
  --configs metaworld \
  --task metaworld_pick_place \
  --agent.dyn.rssm.moe True \
  --agent.grounded.enabled True

# Run vanilla DreamerV3 baseline
MUJOCO_GL=egl python -m dreamerv3.main \
  --configs metaworld \
  --task metaworld_pick_place

# Run tests
python -m pytest tests/ -v
```

## Key Configuration

```yaml
agent:
  dyn:
    rssm:
      moe: True            # Enable MoE dynamics
      num_experts: 4        # Number of homogeneous experts
      balance_coef: 0.01    # Load balancing loss weight
  grounded:
    enabled: True           # Enable TRD + trust weighting
    trd_hidden: 256         # TRD hidden layer size
    tau: 0.15               # Trust threshold
    trust_gamma: 0.5        # Discount factor (0=independent, 1=cumprod)
    trd_scale: 0.0625       # TRD loss scale (1/16, equiv K_trd=16)
    warmup_steps: 5000      # Steps before enabling TRD/DAgger
```

## Toggle Features

| Config | Effect |
|--------|--------|
| `grounded.enabled: False` + `moe: False` | Vanilla DreamerV3 |
| `grounded.enabled: False` + `moe: True` | MoE only (Ablation A2 inverse) |
| `grounded.enabled: True` + `moe: False` | TRD only (Ablation A2) |
| `grounded.enabled: True` + `moe: True` | Full system |

## Citation

```
@article{grounded_imagination_2026,
  title={Grounded Imagination: Learning When and How to Trust World Model
         Rollouts for Contact-Rich Manipulation},
  year={2026}
}
```
