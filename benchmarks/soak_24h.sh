#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build-verify"
soak_root="${build_dir}/soak-24h"
started_epoch="$(date +%s)"
deadline_epoch=$((started_epoch + 86400))
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${soak_root}/${run_id}"
mkdir -p "${run_dir}"

round_number=0
failed_steps=0
interrupted=0

cat > "${run_dir}/manifest.txt" <<MANIFEST
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
minimum_duration_seconds=86400
repository=${repo_root}
build_directory=${build_dir}
initial_commit=$(git -C "${repo_root}" rev-parse HEAD)
MANIFEST

date -u +%Y-%m-%dT%H:%M:%SZ > "${run_dir}/heartbeat.txt"
uname -a > "${run_dir}/system.txt"
git -C "${repo_root}" status --short > "${run_dir}/initial_git_status.txt"

handle_interrupt() {
  interrupted=1
  printf '%s interrupted after %s seconds\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(( $(date +%s) - started_epoch ))" \
    >> "${run_dir}/manifest.txt"
  exit 130
}
trap handle_interrupt INT TERM HUP

run_step() {
  local name="$1"
  shift
  local log_file="${round_dir}/${name}.log"
  printf '%s round=%04d step=%s started\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$round_number" "$name" \
    | tee -a "${round_dir}/status.log"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${run_dir}/heartbeat.txt"
  "$@" > "${log_file}" 2>&1
  local result=$?
  printf '%s round=%04d step=%s exit=%d\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$round_number" "$name" "$result" \
    | tee -a "${round_dir}/status.log"
  if (( result != 0 )); then
    failed_steps=$((failed_steps + 1))
  fi
}

printf '%s run_dir=%s minimum_seconds=86400\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_dir"

while :; do
  round_number=$((round_number + 1))
  round_id="$(printf '%04d' "$round_number")"
  round_dir="${run_dir}/round-${round_id}"
  mkdir -p "${round_dir}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${round_dir}/started_utc.txt"
  git -C "${repo_root}" rev-parse HEAD > "${round_dir}/commit.txt"
  git -C "${repo_root}" diff --binary | sha256sum > "${round_dir}/working_tree_diff.sha256"

  run_step build cmake --build "${build_dir}" -j4
  run_step ctest ctest --test-dir "${build_dir}" --output-on-failure
  run_step api_coverage python3 "${repo_root}/benchmarks/check_api_coverage.py" \
    --build-dir "${build_dir}"
  run_step performance env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "${build_dir}/benchmarks/matx_benchmarks" --full

  date -u +%Y-%m-%dT%H:%M:%SZ > "${round_dir}/finished_utc.txt"
  printf '%s round=%04d complete failed_steps=%d\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$round_number" "$failed_steps" \
    | tee -a "${round_dir}/status.log"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${run_dir}/heartbeat.txt"

  current_epoch="$(date +%s)"
  if (( current_epoch >= deadline_epoch )); then
    elapsed=$((current_epoch - started_epoch))
    {
      printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'elapsed_seconds=%s\n' "$elapsed"
      printf 'completed_rounds=%s\n' "$round_number"
      printf 'failed_steps=%s\n' "$failed_steps"
      printf 'interrupted=%s\n' "$interrupted"
    } >> "${run_dir}/manifest.txt"
    printf '%s soak complete elapsed_seconds=%s rounds=%s failed_steps=%s logs=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$elapsed" "$round_number" \
      "$failed_steps" "$run_dir"
    (( failed_steps == 0 )) && exit 0 || exit 1
  fi
done
