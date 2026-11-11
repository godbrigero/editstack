# Preset: SmolVLA on SO-101 (MuJoCo yellow cube, measured cameras)

    editstack new ~/stacks/smolvla-so101 --weights /path/to/smolvla/checkpoint --link \
        --python /path/to/sim/venv/bin/python --preset smolvla-so101 \
        --task "Generalise yellow-cube picking to cubes 2-6 cm outside the trained band"

Gives you:
- `env/adapter.py`: the measured-camera sim harness, a port of evalpick.py + bench.py. It reproduced the old runs
  bit-for-bit. It provides `capture()` for offline probes and `rescore()`, and saves an observation at every chunk
  boundary.
- `eval/suites`: id / near / heldout / screen, with the original placements at full precision, so results pair with
  the old runs.
- The proven edit chain, with the original torch functions vendored in `env/legacy/`:

      pad0 -> deep8 -> pan17 -> gshr -> cf2      (or -> cf3p25)

  `cf2` and `cf3p25` reproduce the original exports. Their circuit and map tensors are bit-identical; three tensors
  differ at float rounding.
- `model/ARCHITECTURE.md`, pre-filled with verified facts.

The camera maps are not shipped (59 MB each). Copy them in from the original project:

    cp <old project>/agents/camfix/data/maps2.npz  edits/cf2/data/
    cp <old project>/agents/camfix/data/maps3.npz  edits/cf3p25/data/

Set `"sim_repo"` in stack.json (or env `SO101_REPO`) to your copy of the airtruck sim code.
