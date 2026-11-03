#!/usr/bin/env bash
# =====================================================================
# run_extract.sh —— DAG-GCS 数据准备：抽取执行器（只读）
#
# 用法：
#   ./run_extract.sh              # 执行全部抽取
#   ./run_extract.sh mimic_00 ... # 只执行指定前缀的脚本
#
# 机制：每个 sql/*.sql 只含一条 SELECT；本脚本用
#         COPY (<SELECT>) TO STDOUT WITH (FORMAT csv, HEADER true)
#       包裹后重定向到 output/<同名>.csv。
#       这样 SQL 文件保持纯 SQL、可单独审阅，且不依赖 psql 元命令续行。
#
# 安全性：全程只读。不使用 CREATE/INSERT/UPDATE/DELETE/TEMP TABLE，
#         不写入数据库，不改动既有分析文件。
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQLDIR="$HERE/sql"
OUTDIR="$HERE/output"
PGHOST_="-h localhost -U postgres"
PSQL="/Library/PostgreSQL/12/bin/psql"

mkdir -p "$OUTDIR"

# 脚本名 -> 目标数据库
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
  # 去掉行尾分号，便于嵌入 COPY (...)
  local q; q="$(sed -e 's/[[:space:]]*;[[:space:]]*$//' "$sql")"
  # 注意：不可依赖 set -e（本函数在 || 条件上下文中会被禁用），
  #       必须显式检查 psql 退出码，否则 SQL 报错会被静默吞掉。
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

# 选择要跑的脚本
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
