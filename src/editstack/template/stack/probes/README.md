# probes/

Shared, reusable probe scripts (agents keep one-off scripts in agents/<name>/). Run under the stack's python:

    editstack run probes/linear_decodability.py --variant original --obs runs/original/id/s0

A probe writes its numeric output to agents/<name>/data/ or prints it, and its conclusion goes to
model/MODEL_MAP.md with the evidence path. Probes load models only via `Stack.policy()` (RAM slot).
