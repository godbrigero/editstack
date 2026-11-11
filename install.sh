#!/bin/sh
# editstack one-command installer.
#
#   curl -LsSf https://raw.githubusercontent.com/godbrigero/editstack/main/install.sh | sh
#   curl -LsSf https://raw.githubusercontent.com/godbrigero/editstack/main/install.sh | sh -s -- \
#        ~/stacks/my-model --weights /path/to/model --task "what to change" [--preset NAME] [--python PY]
#
# Installs uv if missing, installs/updates the `editstack` CLI from GitHub, and, if arguments are given,
# runs `editstack new <args>` to create a stack. EDITSTACK_REF=<branch|tag|commit> pins a version.
set -eu

REPO="${EDITSTACK_REPO:-https://github.com/godbrigero/editstack}"
REF="${EDITSTACK_REF:-main}"

say() { printf '\033[1m[editstack]\033[0m %s\n' "$*"; }

if ! command -v uv >/dev/null 2>&1; then
    say "uv not found - installing it (https://astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # make uv visible in this shell
    for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        [ -x "$d/uv" ] && PATH="$d:$PATH"
    done
    export PATH
fi

say "installing editstack from $REPO@$REF"
uv tool install --force --python 3.12 "git+$REPO@$REF" >/dev/null

BIN="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
case ":$PATH:" in
    *":$BIN:"*) ;;
    *) PATH="$BIN:$PATH"; export PATH
       say "add $BIN to your PATH (e.g. run: uv tool update-shell)";;
esac

say "installed: $(command -v editstack)"
if [ "$#" -gt 0 ]; then
    editstack new "$@"
else
    say "next: editstack new ~/stacks/my-model --weights /path/to/model --task \"what to change\""
    say "presets: $(editstack presets | awk '{print $1}' | tr '\n' ' ')"
fi
