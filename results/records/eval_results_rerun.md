# Evaluation results, rerun of the evaluation-only route

The 23 evaluations of `eval_queue_hosted.txt`, run from a clean clone of this repository with the four hosted
checkpoints, three at a time, in 17 h 47 min. Same task, seed and rollout count as `../eval_results.md`. Wilson 95 %
interval. VRAM peak is the whole GPU during the run, other jobs included.

| date | label | task | checkpoint | seed | successes | rate | 95% CI | VRAM peak |
|---|---|---|---|---|---|---|---|---|
| 2026-11-08 | p2_v2_600_ep600 | Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0 | p2_model_epoch_600.pth | seed 0 | 38/100 | 0.380 | [0.291, 0.478] | 31147 MiB |
| 2026-11-08 | p3_v2_600_ep600 | Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0 | p3_model_epoch_600.pth | seed 0 | 44/100 | 0.440 | [0.347, 0.538] | 31147 MiB |
| 2026-11-08 | p1_v2_600_ep160 | Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0 | p1_model_epoch_160.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 31105 MiB |
| 2026-11-08 | p4_v2_600_ep600 | Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0 | p4_model_epoch_600.pth | seed 0 | 53/100 | 0.530 | [0.433, 0.625] | 29443 MiB |
| 2026-11-08 | p4_v2_600_shift_camera_offset | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p4_model_epoch_600.pth | seed 0 | 28/100 | 0.280 | [0.201, 0.375] | 29011 MiB |
| 2026-11-08 | p4_v2_600_shift_light_color | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p4_model_epoch_600.pth | seed 0 | 45/100 | 0.450 | [0.356, 0.548] | 24641 MiB |
| 2026-11-09 | p4_v2_600_shift_light_intensity | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p4_model_epoch_600.pth | seed 0 | 24/100 | 0.240 | [0.167, 0.332] | 20800 MiB |
| 2026-11-09 | p4_v2_600_shift_light_texture | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p4_model_epoch_600.pth | seed 0 | 30/100 | 0.300 | [0.219, 0.396] | 20777 MiB |
| 2026-11-09 | p4_v2_600_shift_distractors | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p4_model_epoch_600.pth | seed 0 | 2/100 | 0.020 | [0.006, 0.070] | 20982 MiB |
| 2026-11-09 | p3_v2_600_shift_camera_offset | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p3_model_epoch_600.pth | seed 0 | 22/100 | 0.220 | [0.150, 0.311] | 20801 MiB |
| 2026-11-09 | p3_v2_600_shift_light_color | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p3_model_epoch_600.pth | seed 0 | 37/100 | 0.370 | [0.282, 0.468] | 20777 MiB |
| 2026-11-09 | p3_v2_600_shift_light_intensity | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p3_model_epoch_600.pth | seed 0 | 9/100 | 0.090 | [0.048, 0.162] | 20981 MiB |
| 2026-11-09 | p3_v2_600_shift_light_texture | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p3_model_epoch_600.pth | seed 0 | 28/100 | 0.280 | [0.201, 0.375] | 20992 MiB |
| 2026-11-09 | p3_v2_600_shift_distractors | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p3_model_epoch_600.pth | seed 0 | 25/100 | 0.250 | [0.175, 0.343] | 20992 MiB |
| 2026-11-09 | p2_v2_600_shift_camera_offset | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p2_model_epoch_600.pth | seed 0 | 9/100 | 0.090 | [0.048, 0.162] | 20992 MiB |
| 2026-11-09 | p2_v2_600_shift_light_color | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p2_model_epoch_600.pth | seed 0 | 7/100 | 0.070 | [0.034, 0.137] | 20778 MiB |
| 2026-11-09 | p2_v2_600_shift_light_intensity | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p2_model_epoch_600.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 20982 MiB |
| 2026-11-09 | p2_v2_600_shift_light_texture | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p2_model_epoch_600.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 21346 MiB |
| 2026-11-09 | p2_v2_600_shift_distractors | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p2_model_epoch_600.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 21354 MiB |
| 2026-11-09 | p4_v2_600_shift_combined | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p4_model_epoch_600.pth | seed 0 | 1/100 | 0.010 | [0.002, 0.054] | 21661 MiB |
| 2026-11-09 | p3_v2_600_shift_combined | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p3_model_epoch_600.pth | seed 0 | 1/100 | 0.010 | [0.002, 0.054] | 21663 MiB |
| 2026-11-09 | p2_v2_600_shift_combined | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p2_model_epoch_600.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 21661 MiB |
| 2026-11-09 | p1_v2_600_shift_combined | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | p1_model_epoch_160.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 21659 MiB |
