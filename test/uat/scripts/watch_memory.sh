#!/usr/bin/env bash
# Live tail of the three signals that reveal Honcho+Bonsai memory formation.
#
# usage:
#   bash test/uat/scripts/watch_memory.sh                # all peers
#   bash test/uat/scripts/watch_memory.sh <workspace>    # filter by workspace
#
# Open this in a dedicated pane while you chat with `hermes` in another pane.

set -u

HERMES_HOME="${HERMES_HOME:-$HOME/hermes-stack}"
HONCHO_DIR="$HERMES_HOME/honcho"
WS_FILTER="${1:-}"

printf '\033[1;36m[watch_memory]\033[0m tailing Bonsai + Honcho deriver + documents count\n'
printf '[watch_memory] workspace filter: %s\n' "${WS_FILTER:-(none — showing all)}"
printf '[watch_memory] Ctrl+C to exit\n\n'

# Color helpers (cyan for bonsai, magenta for deriver, yellow for docs)
color_bonsai() { sed 's/^/\o033[36m[bonsai]\o033[0m /'; }
color_deriver() { sed 's/^/\o033[35m[deriver]\o033[0m /'; }
color_docs() { sed 's/^/\o033[33m[docs  ]\o033[0m /'; }

# 1. Bonsai tail — keep only lines that actually reveal inference activity
tail -F -n 0 "$HERMES_HOME/bonsai.log" 2>/dev/null \
  | grep --line-buffered -E 'launch_slot_|prompt processing|print_timing|release:|n_ctx_slot' \
  | color_bonsai &

# 2. Deriver log — observation formation output
docker compose -f "$HONCHO_DIR/docker-compose.yml" logs -f --since=5s deriver 2>/dev/null \
  | grep --line-buffered -E 'PERFORMANCE|Observation|Saved|observation|representation|complete|WARNING|ERROR|Processing|queue|bonsai-8b' \
  | color_deriver &

# 3. Poll documents count every 5s and print a diff line whenever it changes
(
  LAST=-1
  while sleep 5; do
    Q="SELECT COUNT(*) FROM documents"
    [ -n "$WS_FILTER" ] && Q="$Q WHERE workspace_name = '$WS_FILTER'"
    N=$(docker compose -f "$HONCHO_DIR/docker-compose.yml" exec -T database \
        psql -U honcho -d honcho -At -c "$Q;" 2>/dev/null)
    if [ -n "$N" ] && [ "$N" != "$LAST" ]; then
      printf '%s\n' "$(date +%H:%M:%S) documents total = $N (workspace filter: ${WS_FILTER:-all})"
      LAST="$N"
    fi
  done
) | color_docs &

wait
