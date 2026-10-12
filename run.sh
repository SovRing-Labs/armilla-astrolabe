#!/usr/bin/env bash
# Starts Armilla Astrolabe (same as ./install.sh).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/install.sh" "$@"
