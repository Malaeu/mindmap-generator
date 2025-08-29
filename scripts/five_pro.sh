#!/usr/bin/env bash
set -u

# Usage: bash scripts/five_pro.sh python_exe
PYBIN=${1:-python3}
# Detect workers: env FIVE_PRO_WORKERS or nproc
if [ -n "${FIVE_PRO_WORKERS:-}" ]; then
  WORKERS="${FIVE_PRO_WORKERS}"
else
  if command -v nproc >/dev/null 2>&1; then
    WORKERS=$(nproc)
  elif command -v sysctl >/dev/null 2>&1; then
    WORKERS=$(sysctl -n hw.ncpu)
  else
    WORKERS=4
  fi
fi

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="five_pro_${TS}"
CERTS_DIR="${OUTDIR}/certs"
LOGS_DIR="${OUTDIR}/logs"
REPORTS_DIR="${OUTDIR}/reports"
ZIP_NAME="five_pro_bundle_${TS}.zip"

mkdir -p "${CERTS_DIR}" "${LOGS_DIR}" "${REPORTS_DIR}" certs reports || true

PROBLEMS_FILE="${OUTDIR}/PROBLEMS.txt"
echo "# Five-Pro Run Problems" > "${PROBLEMS_FILE}"
echo "Started: ${TS}" >> "${PROBLEMS_FILE}"

run_job() {
  local family="$1"; shift
  local grid="$1"; shift
  local zeros="$1"; shift
  local pcut="$1"; shift
  local amain="$1"; shift
  local ptailm="$1"; shift
  local ztailm="$1"; shift

  local tag="${family// /_}"
  local json_path="${CERTS_DIR}/${tag}.json"
  local log_path="${LOGS_DIR}/${tag}.log"

  echo "[RUN] family=${family} grid=${grid} zeros=${zeros} pcut=${pcut} a-main=${amain}" | tee -a "${log_path}"
  CMD=("${PYBIN}" -m experiments.certified_validation \
    --family "${family}" \
    --zeros "${zeros}" \
    --grid "${grid}" \
    --p-cut "${pcut}" \
    --a-main "${amain}" \
    --p-tail-m "${ptailm}" \
    --z-tail-m "${ztailm}" \
    --workers "${WORKERS}" \
    --json "${json_path}")

  { time "${CMD[@]}" ; } >> "${log_path}" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[PROBLEM] family=${family} exited with code ${rc}. See ${log_path}" | tee -a "${PROBLEMS_FILE}"
  else
    echo "[OK] family=${family} -> ${json_path}" | tee -a "${log_path}"
  fi
}

# Strong parameter presets per family
run_job gaussian  "1.0,2.0,5.0,10.0" 1000 10000 3000 1500 1500
run_job cauchy    "1.0,2.0,3.0"      1000 10000 3000 1500 1500
run_job bump      "1.0,2.0,3.0"      1000  8000 2500 1200 1200
run_job autocorr  "1.0,2.0,5.0"      1000 10000 3000 1500 1500

# Copy certs into repo-level certs for exporter compatibility
cp -f ${CERTS_DIR}/*.json certs/ 2>/dev/null || true

# Export markdown summary
${PYBIN} -m experiments.export_certificates "${REPORTS_DIR}/CERTIFICATES.md" > "${LOGS_DIR}/export.log" 2>&1 || \
  echo "[PROBLEM] export failed, see ${LOGS_DIR}/export.log" | tee -a "${PROBLEMS_FILE}"

# Also copy exported report to repo reports dir for convenience
cp -f "${REPORTS_DIR}/CERTIFICATES.md" reports/ 2>/dev/null || true

# Create the final zip bundle
zip -r "${ZIP_NAME}" "${OUTDIR}" >/dev/null 2>&1 || {
  echo "[PROBLEM] zip failed" | tee -a "${PROBLEMS_FILE}"
}

echo "Bundle ready: ${ZIP_NAME}"
echo "Problems (if any) recorded in: ${PROBLEMS_FILE}"
