#!/bin/bash
# Launch 8 parallel MetaWorld experiments: Grounded vs Baseline
# Tasks by difficulty: reach (easy) → push (medium) → door-open (medium) → pick-place (hard)
# Usage: bash scripts/run_metaworld.sh

cd /jushen-shanghai/yuxiao/WM/grounded-imagination
export MUJOCO_GL=egl
BASE=/jushen-shanghai/yuxiao/WM/logs
mkdir -p $BASE

# Grounded Imagination (MoE + TRD)
CUDA_VISIBLE_DEVICES=0 nohup python -m dreamerv3.main --configs metaworld --task metaworld_reach --seed 0 --logdir $BASE/grounded_reach_s0 --agent.dyn.rssm.moe True --agent.grounded.enabled True > $BASE/log_g_reach_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=1 nohup python -m dreamerv3.main --configs metaworld --task metaworld_push --seed 0 --logdir $BASE/grounded_push_s0 --agent.dyn.rssm.moe True --agent.grounded.enabled True > $BASE/log_g_push_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=2 nohup python -m dreamerv3.main --configs metaworld --task metaworld_door_open --seed 0 --logdir $BASE/grounded_door_open_s0 --agent.dyn.rssm.moe True --agent.grounded.enabled True > $BASE/log_g_door_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=3 nohup python -m dreamerv3.main --configs metaworld --task metaworld_pick_place --seed 0 --logdir $BASE/grounded_pick_place_s0 --agent.dyn.rssm.moe True --agent.grounded.enabled True > $BASE/log_g_pp_s0.txt 2>&1 &
sleep 5

# Vanilla DreamerV3 baseline (no MoE, no TRD)
CUDA_VISIBLE_DEVICES=4 nohup python -m dreamerv3.main --configs metaworld --task metaworld_reach --seed 0 --logdir $BASE/baseline_reach_s0 > $BASE/log_b_reach_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=5 nohup python -m dreamerv3.main --configs metaworld --task metaworld_push --seed 0 --logdir $BASE/baseline_push_s0 > $BASE/log_b_push_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=6 nohup python -m dreamerv3.main --configs metaworld --task metaworld_door_open --seed 0 --logdir $BASE/baseline_door_open_s0 > $BASE/log_b_door_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=7 nohup python -m dreamerv3.main --configs metaworld --task metaworld_pick_place --seed 0 --logdir $BASE/baseline_pick_place_s0 > $BASE/log_b_pp_s0.txt 2>&1 &

echo "All 8 MetaWorld experiments launched."
echo "  GPU 0-3: Grounded (reach, push, door-open, pick-place)"
echo "  GPU 4-7: Baseline (reach, push, door-open, pick-place)"
echo "Monitor: nvidia-smi"
