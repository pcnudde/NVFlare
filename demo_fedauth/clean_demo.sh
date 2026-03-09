#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

remove_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    return
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'would remove %s\n' "$path"
    return
  fi

  rm -rf "$path"
  printf 'removed %s\n' "$path"
}

cd "$SCRIPT_DIR"

if command -v podman >/dev/null 2>&1; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "would run: podman compose down --remove-orphans"
  else
    podman compose down --remove-orphans >/dev/null 2>&1 || true
  fi
fi

remove_path "$SCRIPT_DIR/workspace"
remove_path "$SCRIPT_DIR/distribution"
remove_path "$SCRIPT_DIR/user_demo"
remove_path "$SCRIPT_DIR/presentation/node_modules"
remove_path "$SCRIPT_DIR/presentation/fedauth_demo_slides.html"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "dry run complete"
else
  echo "demo_fedauth cleanup complete"
fi
