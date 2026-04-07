#!/bin/bash
# MoE-only ablation: isolate MoE contribution from TRD
# GPU 0-3: MoE only, GPU 4-7: Baseline (fresh run for fair comparison)
# Usage: bash scripts/run_moe_only.sh

cd /jushen-shanghai/yuxiao/WM/grounded-imagination
export MUJOCO_GL=egl
BASE=/jushen-shanghai/yuxiao/WM/logs
mkdir -p $BASE

# MoE only (no TRD, no trust weighting)
CUDA_VISIBLE_DEVICES=0 nohup python -m dreamerv3.main --configs metaworld --task metaworld_pick_place --seed 0 --logdir $BASE/moe_only_pick_place_s0 --agent.dyn.rssm.moe True --agent.grounded.enabled False > $BASE/log_m_pp_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=1 nohup python -m dreamerv3.main --configs metaworld --task metaworld_door_open --seed 0 --logdir $BASE/moe_only_door_open_s0 --agent.dyn.rssm.moe True --agent.grounded.enabled False > $BASE/log_m_door_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=2 nohup python -m dreamerv3.main --configs metaworld --task metaworld_push --seed 0 --logdir $BASE/moe_only_push_s0 --agent.dyn.rssm.moe True --agent.grounded.enabled False > $BASE/log_m_push_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=3 nohup python -m dreamerv3.main --configs metaworld --task metaworld_reach --seed 0 --logdir $BASE/moe_only_reach_s0 --agent.dyn.rssm.moe True --agent.grounded.enabled False > $BASE/log_m_reach_s0.txt 2>&1 &
sleep 5

# Baseline (fresh run, same code version for fair comparison)
CUDA_VISIBLE_DEVICES=4 nohup python -m dreamerv3.main --configs metaworld --task metaworld_pick_place --seed 0 --logdir $BASE/baseline2_pick_place_s0 > $BASE/log_b2_pp_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=5 nohup python -m dreamerv3.main --configs metaworld --task metaworld_door_open --seed 0 --logdir $BASE/baseline2_door_open_s0 > $BASE/log_b2_door_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=6 nohup python -m dreamerv3.main --configs metaworld --task metaworld_push --seed 0 --logdir $BASE/baseline2_push_s0 > $BASE/log_b2_push_s0.txt 2>&1 &
sleep 5
CUDA_VISIBLE_DEVICES=7 nohup python -m dreamerv3.main --configs metaworld --task metaworld_reach --seed 0 --logdir $BASE/baseline2_reach_s0 > $BASE/log_b2_reach_s0.txt 2>&1 &

echo "All 8 experiments launched."
echo "  GPU 0-3: MoE-only (reach, push, door-open, pick-place)"
echo "  GPU 4-7: Baseline fresh (same 4 tasks)"
echo "Monitor: nvidia-smi"
echo "Results: for f in $BASE/log_m_*.txt $BASE/log_b2_*.txt; do echo \$(basename \$f) && grep episode/score \$f | tail -1; done"
