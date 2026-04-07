#!/bin/bash
# MoE-only ablation: isolate MoE contribution from TRD
# Compare MoE-only vs Baseline on tasks where Grounded underperformed
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

echo "MoE-only experiments launched on GPU 0-3."
echo "GPU 4-7 still running previous experiments (if any)."
echo "Monitor: nvidia-smi"
echo "Results: grep episode/score $BASE/log_m_*.txt | tail -4"
