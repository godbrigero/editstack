# Phase-0 harness audit (complete before any edit)

Tick each item and write the evidence (command, file, number) next to it. Copy the result into
model/ARCHITECTURE.md section 5. Bump `harness_version` in stack.json whenever anything here changes later.

## Model I/O
- [ ] Weights load in the adapter from `tensors` and reproduce the original model's output on one recorded input
      (bit-equal or float tolerance). Evidence:
- [ ] Preprocessing identical to deployment (resize/pad, normalisation stats file, tokenizer, prompt string).
- [ ] Output post-processing identical (de-normalisation, clipping, which channels are used).
- [ ] Inference loop identical (sampling steps, noise, chunk length, how many steps execute before re-planning).

## Simulation vs deployment
- [ ] Cameras: pose, FOV, resolution, which stream (e.g. RGB not IR), mounting, measured on the real rig. ASK THE
      USER ON DAY 1 for one real frame per camera, and compare it side by side with the sim render of the same scene.
      (A mismatch found on day 4 collapsed the best variant to 6/40 and forced a camera-transfer stage.)
- [ ] Calibration of joints/units, control rate, latency.
- [ ] Scene: object size/colour/mass, table height, lighting, shadows (the real world may have none).
- [ ] The sim code being run is the CURRENT one (not a stale copy). Record its path and version/commit.

## Benchmark
- [ ] Success definition agreed with the user, written in stack.json metrics, staged (e.g. reach, pick).
- [ ] Outcome codes for each failure type (short / long / wide / too high / no attempt / dropped ...).
- [ ] Thresholds calibrated on past or baseline episodes (e.g. 99.7% of picks close within 1.5 cm).
- [ ] Suites: screen, id, ood, heldout with fixed episode lists; heldout never used for selection.
- [ ] Determinism: same (variant, seed, episode) twice -> same result. If porting an older harness, reproduce one
      old episode bit-for-bit (store episode parameters at full float precision: rounding initial poses to 6
      decimals changed closed-loop results by ~2% when this stack was validated).
- [ ] Baseline on the board: original on id (2 seeds) and ood (3 seeds).
- [ ] Episode wall time measured (sets the time budget) and RAM per process measured (`footprint`/`/usr/bin/time -l`),
      written into stack.json resources.per_process_gb.

## Trust
- [ ] Every scoring script read and understood (a past "scorer" only counted its own invocations).
- [ ] Real-hardware rules agreed with the user (stack.json real_hardware).
