#!/usr/bin/env bash
# =====================================================================
# run_extract.sh —— MIMIC 临床解释探索：抽取执行器（只读）
#
# 与 dag_gcs_next_stage/run_extract.sh 同构，但**输出到本目录的 output/**，
# 不改动、不覆盖上游交付的任何文件。
#
# 用法：
#   ./run_extract.sh              # 执行本目录 sql/ 下全部脚本
#   ./run_extract.sh mimic_12 ... # 只执行指定脚本
#
# 安全性：全程只读。不使用 CREATE/INSERT/UPDATE/DELETE/TEMP TABLE。
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQLDIR="$HERE/sql"
OUTDIR="$HERE/output"
PGHOST_="-h localhost -U postgres"
PSQL="/Library/PostgreSQL/12/bin/psql"

mkdir -p "$OUTDIR"

db_for() {
  case "$1" in
    mimic_*) echo "mimiciv31" ;;
    eicu_*)  echo "eicu" ;;
    *) echo "ERROR: 未知前缀 $1" >&2; exit 1 ;;
  esac
}

run_one() {
  local name="$1"
  local db; db=$(db_for "$name")
  local sql="$SQLDIR/$name.sql"
  local out="$OUTDIR/$name.csv"
  if [[ ! -f "$sql" ]]; then echo "skip (no file): $name"; return 0; fi
  printf '%-36s -> %-10s ' "$name" "$db"
  local t0=$SECONDS
  local q; q="$(sed -e 's/[[:space:]]*;[[:space:]]*$//' "$sql")"
  local rc=0
  "$PSQL" $PGHOST_ -d "$db" -v ON_ERROR_STOP=1 -q \
      -c "COPY ( $q ) TO STDOUT WITH (FORMAT csv, HEADER true, NULL '')" \
      > "$out" 2> "$out.err" || rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAILED (psql rc=$rc)"
    sed -e 's/^/    /' "$out.err" | head -20
    rm -f "$out" "$out.err"
    return 1
  fi
  rm -f "$out.err"
  local n=$(( $(wc -l < "$out") - 1 ))
  local dt=$(( SECONDS - t0 ))
  echo "rows=${n}  ${dt}s"
}

if [[ $# -gt 0 ]]; then
  names=("$@")
else
  names=()
  while IFS= read -r f; do
    names+=("$(basename "$f" .sql)")
  done < <(ls "$SQLDIR"/*.sql | sort)
fi

for n in "${names[@]}"; do
  run_one "$n" || { echo "FAILED: $n" >&2; exit 1; }
done

echo "extract complete -> $OUTDIR"
