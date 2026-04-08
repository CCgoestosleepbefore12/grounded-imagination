#!/bin/bash
# Controlled experiment: MoE-only vs Baseline on bring_ball
# DreamerV3 official dmc_vision config (size200m, imag=15)
# bring_ball: DreamerV3 officially scores 0 on this task
# Usage: bash scripts/run_bring_ball.sh

cd /jushen-shanghai/yuxiao/WM/grounded-imagination
export MUJOCO_GL=egl
BASE=/jushen-shanghai/yuxiao/WM/logs

# MoE-only (official config + MoE, the only change)
CUDA_VISIBLE_DEVICES=0 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 0 --logdir $BASE/moe_bring_ball_s0 --agent.dyn.rssm.moe True --run.envs 8 > $BASE/log_moe_bb_s0.txt 2>&1 &
sleep 10
CUDA_VISIBLE_DEVICES=1 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 1 --logdir $BASE/moe_bring_ball_s1 --agent.dyn.rssm.moe True --run.envs 8 > $BASE/log_moe_bb_s1.txt 2>&1 &
sleep 10
CUDA_VISIBLE_DEVICES=2 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 2 --logdir $BASE/moe_bring_ball_s2 --agent.dyn.rssm.moe True --run.envs 8 > $BASE/log_moe_bb_s2.txt 2>&1 &
sleep 10

# Baseline (pure DreamerV3 official config)
CUDA_VISIBLE_DEVICES=3 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 0 --logdir $BASE/base_bring_ball_s0 --run.envs 8 > $BASE/log_base_bb_s0.txt 2>&1 &
sleep 10
CUDA_VISIBLE_DEVICES=4 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 1 --logdir $BASE/base_bring_ball_s1 --run.envs 8 > $BASE/log_base_bb_s1.txt 2>&1 &
sleep 10
CUDA_VISIBLE_DEVICES=5 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 2 --logdir $BASE/base_bring_ball_s2 --run.envs 8 > $BASE/log_base_bb_s2.txt 2>&1 &

echo "6 experiments launched (3 MoE + 3 Baseline, 3 seeds each)"
echo "  GPU 0-2: MoE-only bring_ball seed 0,1,2"
echo "  GPU 3-5: Baseline bring_ball seed 0,1,2"
echo "  Config: dmc_vision (size200m, imag=15, train_ratio=256, envs=8)"
echo "  Only difference: --agent.dyn.rssm.moe True"
echo ""
echo "Check: for f in \$BASE/log_moe_bb_*.txt \$BASE/log_base_bb_*.txt; do echo \$(basename \$f) && grep episode/score \$f | tail -1; done"
