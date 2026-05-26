# Repository Instructions

- Keep the current L7 experiment setup, run naming, dataset choices, and training parameters documented in `README_l7.md`.
- Before changing any documented L7 experiment setup or parameter setting, explicitly ask the user for approval first.
- Before launching L7 training or eval/video jobs, check active `train.py`/`play_l7.py` processes, `nvidia-smi`, and cgroup memory usage/limit from `/sys/fs/cgroup/memory/memory.usage_in_bytes` and `/sys/fs/cgroup/memory/memory.limit_in_bytes`.
- Treat the 50 GB cgroup memory limit as a hard bottleneck, not only GPU memory. Leave at least 10 GB cgroup headroom before starting eval/video or another Isaac Gym job.
- Do not start many Isaac Gym L7 jobs at the same time. Start single-class jobs sequentially, and wait until each one reaches `Learning iteration 0` before launching the next.
- Do not run `eval_l7.sh` video generation while seven 1024-env single-class jobs are active. Stop or reduce training jobs first, and state clearly when a run was intentionally terminated to free resources instead of calling it a training failure.
- Use the observed memory profile as a guardrail: each 1024-env L7 single-class run is roughly 5.4 GB RSS and 6 GB GPU memory; seven such runs plus eval/video can push the cgroup near the 50 GB limit even when GPU memory still looks available.
