#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="experiments/e1"
RUNNER="experiments_scripts/run_patchcore.py"
DIAGNOSTIC="experiments_scripts/compute_e1_diagnostics.py"

CLASSES=(
  "rice"
  "can"
  "wallplugs"
  "sheet_metal"
)

MODES=(
  "baseline"
  "identity"
  "adapted"
)

declare -A CONFIG_PREFIX=(
  ["baseline"]="r0_baseline"
  ["identity"]="r0_identity"
  ["adapted"]="r1_adapted"
)

declare -A EXPERIMENT_NAME=(
  ["baseline"]="R0_baseline"
  ["identity"]="R0_identity"
  ["adapted"]="R1_adapted"
)

mkdir -p "$OUTPUT_ROOT"

latest_completed_run() {
  local class_name="$1"
  local mode="$2"
  local experiment_name="${EXPERIMENT_NAME[$mode]}"
  local candidate

  while IFS= read -r -d '' candidate; do
    if [[ -f "$candidate/metrics.json" && -f "$candidate/predictions.npz" ]]; then
      if [[ "$mode" != "adapted" || -f "$candidate/adapter_training.json" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done < <(
    find "$OUTPUT_ROOT" -maxdepth 1 -type d \
      -name "${experiment_name}_${class_name}_*" \
      -printf '%T@ %p\0' \
      | sort -z -nr \
      | cut -z -d ' ' -f 2-
  )

  return 1
}

echo "========================================"
echo "Running E1 PatchCore experiments"
echo "Classes: ${CLASSES[*]}"
echo "Modes:   ${MODES[*]}"
echo "Output:  $OUTPUT_ROOT"
echo "========================================"

for class_name in "${CLASSES[@]}"; do
  for mode in "${MODES[@]}"; do
    config="configs/mvtec2_experiments/${CONFIG_PREFIX[$mode]}_mvtec_ad2_${class_name}.yaml"

    echo
    echo "----------------------------------------"
    echo "Class : $class_name"
    echo "Mode  : $mode"
    echo "Config: $config"
    echo "----------------------------------------"

    if [[ ! -f "$config" ]]; then
      echo "ERROR: Config not found: $config" >&2
      exit 1
    fi

    if completed_run="$(latest_completed_run "$class_name" "$mode")"; then
      echo "SKIP: Completed run exists: $completed_run"
      continue
    fi

    if [[ "${E1_DRY_RUN:-0}" == "1" ]]; then
      echo "DRY_RUN: Would run $config"
      continue
    fi

    python -u "$RUNNER" \
      --config "$config" \
      --output-root "$OUTPUT_ROOT"
  done
done

if [[ "${E1_DRY_RUN:-0}" == "1" ]]; then
  echo
  echo "========================================"
  echo "DRY_RUN: Skipping diagnostics"
  echo "========================================"
  exit 0
fi

echo
echo "========================================"
echo "Running diagnostics"
echo "========================================"

prediction_count=0

while IFS= read -r -d '' predictions; do
  run_dir="$(dirname "$predictions")"
  output_json="$run_dir/e1_diagnostics.json"

  echo
  echo "Diagnostic: $predictions"

  python "$DIAGNOSTIC" \
    --predictions "$predictions" \
    --output "$output_json"

  prediction_count=$((prediction_count + 1))
done < <(find "$OUTPUT_ROOT" -type f -name "predictions.npz" -print0)

if [[ "$prediction_count" -eq 0 ]]; then
  echo "WARNING: No predictions.npz found under $OUTPUT_ROOT" >&2
  exit 1
fi

echo
echo "========================================"
echo "Completed all experiments and diagnostics"
echo "Diagnostics generated: $prediction_count"
echo "========================================"
