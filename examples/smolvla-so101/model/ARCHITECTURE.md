# Architecture: SmolVLA (LeRobot) on SO-101, yellow cube

Facts carried over from the 2026-11 project (smolvla-weight-edit-6000/MODEL_MAP.md). Re-verify any line you
build on with the stated method; mark new facts [verified: how].

## 1. Inputs
- 2 RGB cameras (front, wrist) -> SigLIP (12 layers, 768-d) -> 64 tokens each via pixel-shuffle + connector
  (12288 -> 960), x sqrt(960). [verified: model_info.md / code]
- Task string "Pick up the yellow cube." -> 48 tokens. Robot state: 6 joints normalised with
  policy_preprocessor_step_5_normalizer stats, padded to 32 -> state_proj (960x32) -> 1 token. [verified: code]
- Noisy action chunk 50x32 (6 real + 26 padding) -> action_in_proj (720x32), fused with 720 sin/cos time
  features in action_time_mlp_in (720x1440). [verified: code]

## 2. Outputs
- action_out_proj (32x720) -> velocity; 10 Euler steps x <- x - 0.1 v, tau 1 -> 0; channels 0-5 used,
  de-normalised with policy_postprocessor_step_0_unnormalizer stats. [verified: code]

## 3. Training objective
- Flow matching, MSE on channels 0-5 only (losses[:, :, :6]); padding channels never supervised but fed noise at
  inference (fixed by edit pad0). [verified: code + jitter probe]
- In-distribution envelope: cube |x| <= 7 cm, y in [-0.28, -0.16] m, yaw +-60 deg.

## 4. Data flow
- VLM (16 layers, 960-d) runs once on the prefix; the action expert (16 layers, 720-d) attends to its K/V:
  even layers joint attention, odd layers cross attention. Images reach actions only through the odd layers.
  [verified: ablation probes, MODEL_MAP]
- Expert layernorm weights are all exactly 1 (bf16) - never moved in training. [verified: decode SCAN.md]
- Cube position is linearly decodable everywhere (R^2 0.87-0.99): failures are read-out, not perception.
- Per-chunk correction gains: pan 0.34-0.6, radial ~0.13, wrist roll ~0.03. First move aimed by the front camera
  alone, reading forward distance not radius (side cubes close short). 49/69 grip failures = roll > 25 deg off.

## 5. Execution at deployment / sim vs real
- 50-step chunks executed fully at 30 Hz, then re-plan. Re-planning every 10/25 steps was worse (9/24, 15/24
  near-lateral picks vs 21/24) and was reverted.
- Sim cameras: MEASURED set (SO101_SIM_CAMERAS=measured, repo of 2026-11-09). The real front camera stream must
  be RGB (it was once the IR stream). Real world has no arm shadow. harness_version measured-cams-v1.
- Episode: 18 s sim, ~33 s wall unloaded (median 50 s loaded); ~9 GB real footprint per process.

## 6. Exact identities
- Action-space translation and per-axis affine scaling: env/smolvla_algebra.py (translate_actions, scale_axis).
- Exact product in expert layer-15 SwiGLU unit pairs (editstack.ops.exact_product_pair), gated on
  planned - current state.
