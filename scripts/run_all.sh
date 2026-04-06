#!/bin/bash
# Launch 8 parallel DMC experiments: Grounded vs Baseline
# cup_catch (easy) + manipulator_bring_ball (hard), multiple seeds
# Usage: bash scripts/run_all.sh

cd /jushen-shanghai/yuxiao/WM/grounded-imagination
export MUJOCO_GL=egl
BASE=/jushen-shanghai/yuxiao/WM/logs
mkdir -p $BASE

# Grounded Imagination (MoE + TRD)
CUDA_VISIBLE_DEVICES=0 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_cup_catch --seed 0 --logdir $BASE/grounded_cup_catch_s0 --agent.dyn.rssm.moe True --agent.grounded.enabled True --run.steps 1.1e6 --run.train_ratio 512 > $BASE/log_g_cup_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=1 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 0 --logdir $BASE/grounded_bring_ball_s0 --agent.dyn.rssm.moe True --agent.grounded.enabled True --run.steps 1.1e6 --run.train_ratio 512 > $BASE/log_g_bb_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=2 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 1 --logdir $BASE/grounded_bring_ball_s1 --agent.dyn.rssm.moe True --agent.grounded.enabled True --run.steps 1.1e6 --run.train_ratio 512 > $BASE/log_g_bb_s1.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=3 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 2 --logdir $BASE/grounded_bring_ball_s2 --agent.dyn.rssm.moe True --agent.grounded.enabled True --run.steps 1.1e6 --run.train_ratio 512 > $BASE/log_g_bb_s2.txt 2>&1 &
sleep 5

# Vanilla DreamerV3 baseline (no MoE, no TRD)
CUDA_VISIBLE_DEVICES=4 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_cup_catch --seed 0 --logdir $BASE/baseline_cup_catch_s0 --run.steps 1.1e6 --run.train_ratio 512 > $BASE/log_b_cup_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=5 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 0 --logdir $BASE/baseline_bring_ball_s0 --run.steps 1.1e6 --run.train_ratio 512 > $BASE/log_b_bb_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=6 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 1 --logdir $BASE/baseline_bring_ball_s1 --run.steps 1.1e6 --run.train_ratio 512 > $BASE/log_b_bb_s1.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=7 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 2 --logdir $BASE/baseline_bring_ball_s2 --run.steps 1.1e6 --run.train_ratio 512 > $BASE/log_b_bb_s2.txt 2>&1 &

echo "All 8 experiments launched."
echo "  GPU 0: Grounded cup_catch s0"
echo "  GPU 1-3: Grounded bring_ball s0,1,2"
echo "  GPU 4: Baseline cup_catch s0"
echo "  GPU 5-7: Baseline bring_ball s0,1,2"
echo "Monitor: nvidia-smi"
