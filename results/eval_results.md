# Evaluation results

In-distribution unless the label says otherwise. Wilson 95% interval. VRAM peak is the whole GPU during the run, other jobs included.

| date | label | task | checkpoint | seed | successes | rate | 95% CI | VRAM peak |
|---|---|---|---|---|---|---|---|---|
| 2026-10-29 | p2_v2_600_ep600 | Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0 | model_epoch_600.pth | seed 0 | 37/100 | 0.370 | [0.282, 0.468] | 31058 MiB |
| 2026-10-29 | p4_v2_600_ep600 | Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0 | model_epoch_600.pth | seed 0 | 58/100 | 0.580 | [0.482, 0.672] | 32006 MiB |
| 2026-10-29 | p3_v2_600_ep600 | Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0 | model_epoch_600.pth | seed 0 | 46/100 | 0.460 | [0.366, 0.557] | 32006 MiB |
| 2026-10-29 | p1_v2_600_ep160 | Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0 | model_epoch_160.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 29247 MiB |
| 2026-10-29 | p4_v2_600_shift_camera_offset | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 28/100 | 0.280 | [0.201, 0.375] | 24530 MiB |
| 2026-10-29 | p4_v2_600_shift_light_color | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 49/100 | 0.490 | [0.394, 0.587] | 24528 MiB |
| 2026-10-29 | p4_v2_600_shift_light_intensity | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 26/100 | 0.260 | [0.184, 0.354] | 20853 MiB |
| 2026-10-29 | p4_v2_600_shift_light_texture | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 32/100 | 0.320 | [0.237, 0.417] | 20651 MiB |
| 2026-10-29 | p4_v2_600_shift_distractors | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 4/100 | 0.040 | [0.016, 0.098] | 20649 MiB |
| 2026-10-29 | p3_v2_600_shift_camera_offset | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 9/100 | 0.090 | [0.048, 0.162] | 20166 MiB |
| 2026-10-29 | p3_v2_600_shift_light_color | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 33/100 | 0.330 | [0.246, 0.427] | 20639 MiB |
| 2026-10-29 | p3_v2_600_shift_light_intensity | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 16/100 | 0.160 | [0.101, 0.244] | 20833 MiB |
| 2026-10-29 | p3_v2_600_shift_light_texture | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 35/100 | 0.350 | [0.264, 0.447] | 20841 MiB |
| 2026-10-29 | p3_v2_600_shift_distractors | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 19/100 | 0.190 | [0.125, 0.278] | 20833 MiB |
| 2026-10-29 | p2_v2_600_shift_camera_offset | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 20/100 | 0.200 | [0.133, 0.289] | 20115 MiB |
| 2026-10-29 | p2_v2_600_shift_light_color | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 5/100 | 0.050 | [0.022, 0.112] | 20627 MiB |
| 2026-10-29 | p2_v2_600_shift_light_intensity | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 20627 MiB |
| 2026-10-29 | p2_v2_600_shift_light_texture | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 20833 MiB |
| 2026-10-29 | p2_v2_600_shift_distractors | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 20831 MiB |
| 2026-11-02 | p4_v2_600_shift_combined | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 21543 MiB |
| 2026-11-02 | p3_v2_600_shift_combined | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 2/100 | 0.020 | [0.006, 0.070] | 21594 MiB |
| 2026-11-02 | p2_v2_600_shift_combined | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_600.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 21543 MiB |
| 2026-11-02 | p1_v2_600_shift_combined | CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 | model_epoch_160.pth | seed 0 | 0/100 | 0.000 | [0.000, 0.037] | 21296 MiB |
