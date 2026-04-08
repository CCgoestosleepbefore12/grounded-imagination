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

# Baseline (pure DreamerV3 official config)
CUDA_VISIBLE_DEVICES=1 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 0 --logdir $BASE/base_bring_ball_s0 --run.envs 8 > $BASE/log_base_bb_s0.txt 2>&1 &

echo "2 experiments launched (GPU 0: MoE, GPU 1: Baseline)"
echo "Config: dmc_vision (size200m, imag=15, train_ratio=256, envs=8)"
echo "Only difference: --agent.dyn.rssm.moe True"
echo "Check: grep episode/score $BASE/log_moe_bb_s0.txt | tail -1 && grep episode/score $BASE/log_base_bb_s0.txt | tail -1"
