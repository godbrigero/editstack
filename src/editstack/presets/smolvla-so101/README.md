# Preset: SmolVLA on SO-101 (MuJoCo yellow cube, measured cameras)

    editstack new ~/stacks/smolvla-so101 \
        --weights ~/Documents/smolvla-weight-edit-6000/ckpt_base --link \
        --python ~/Documents/sentinel-drone-arm-project/.venv/bin/python \
        --preset smolvla-so101 \
        --task "Generalise yellow-cube picking to cubes 2-6 cm outside the trained band"

Gives: the sim adapter (port of evalpick.py + bench.py), suites id/near/heldout/screen with the original
placements (paired with the old runs), the proven edit chain pad0 -> deep8 -> pan17, and pre-filled
ARCHITECTURE.md. Set SO101_REPO if the sim code lives elsewhere.
