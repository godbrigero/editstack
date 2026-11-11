# Edit stack

One folder = one editable model. Put the model in `weights/original/` (or `editstack add-weights PATH`), run
`editstack decode`, implement `env/adapter.py` and the suites in `eval/suites/`, then hand the task to an agent:

> Read AGENTS.md and do Phase 0 for this stack. Task: <what you want changed>.

Status at any time: `editstack status`. Leaderboard: `board/LEADERBOARD.md`. Deployed weights: `weights/best`.
