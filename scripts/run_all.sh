#!/bin/bash
# Launch 8 parallel experiments: Grounded vs Baseline
# DMC cup_catch (easy) + DMC manipulator_bring_ball (hard) + MetaWorld pick-place (hard)
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
CUDA_VISIBLE_DEVICES=2 nohup python -m dreamerv3.main --configs metaworld --task metaworld_pick_place --seed 0 --logdir $BASE/grounded_pick_place_s0 --agent.dyn.rssm.moe True --agent.grounded.enabled True > $BASE/log_g_pp_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=3 nohup python -m dreamerv3.main --configs metaworld --task metaworld_pick_place --seed 1 --logdir $BASE/grounded_pick_place_s1 --agent.dyn.rssm.moe True --agent.grounded.enabled True > $BASE/log_g_pp_s1.txt 2>&1 &
sleep 5

# Vanilla DreamerV3 baseline (no MoE, no TRD)
CUDA_VISIBLE_DEVICES=4 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_cup_catch --seed 0 --logdir $BASE/baseline_cup_catch_s0 --run.steps 1.1e6 --run.train_ratio 512 > $BASE/log_b_cup_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=5 nohup python -m dreamerv3.main --configs dmc_vision --task dmc_manipulator_bring_ball --seed 0 --logdir $BASE/baseline_bring_ball_s0 --run.steps 1.1e6 --run.train_ratio 512 > $BASE/log_b_bb_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=6 nohup python -m dreamerv3.main --configs metaworld --task metaworld_pick_place --seed 0 --logdir $BASE/baseline_pick_place_s0 > $BASE/log_b_pp_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=7 nohup python -m dreamerv3.main --configs metaworld --task metaworld_pick_place --seed 1 --logdir $BASE/baseline_pick_place_s1 > $BASE/log_b_pp_s1.txt 2>&1 &

echo "All 8 experiments launched."
echo "  GPU 0: Grounded cup_catch"
echo "  GPU 1: Grounded bring_ball"
echo "  GPU 2-3: Grounded pick-place seed 0,1"
echo "  GPU 4: Baseline cup_catch"
echo "  GPU 5: Baseline bring_ball"
echo "  GPU 6-7: Baseline pick-place seed 0,1"
echo "Monitor: nvidia-smi"
