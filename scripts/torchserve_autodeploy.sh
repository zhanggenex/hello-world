#!/usr/bin/env bash

set -euo pipefail

# TorchServe auto-deploy script
# - Creates an isolated venv
# - Installs torch/torchvision/torchserve/tooling
# - Builds a TorchScript ResNet-18 and archives to MAR
# - Starts TorchServe and runs a test inference
#
# Usage examples:
#   ./torchserve_autodeploy.sh deploy       # Full flow: setup -> start -> test
#   ./torchserve_autodeploy.sh setup        # Just install deps and build MAR
#   ./torchserve_autodeploy.sh start        # Start TorchServe with built MAR
#   ./torchserve_autodeploy.sh test         # Run a sample inference
#   ./torchserve_autodeploy.sh stop         # Stop TorchServe
#   ./torchserve_autodeploy.sh cleanup      # Remove build artifacts and venv
#   ./torchserve_autodeploy.sh status       # Check server/model status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/workspace"
WORK_DIR="${ROOT_DIR}/torchserve_autodeploy"
VENV_DIR="${WORK_DIR}/.venv"
MODEL_STORE_DIR="${WORK_DIR}/model_store"
TMP_DIR="${WORK_DIR}/tmp"

MODEL_NAME="resnet18"
MODEL_VERSION="1.0"
SERIALIZED_MODEL_PATH="${WORK_DIR}/${MODEL_NAME}_scripted.pt"
MAR_PATH="${MODEL_STORE_DIR}/${MODEL_NAME}.mar"
HANDLER_NAME="image_classifier"

INFERENCE_PORT="8080"
MANAGEMENT_PORT="8081"
METRICS_PORT="8082"

TEST_IMAGE_URL_DEFAULT="https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
TEST_IMAGE_PATH="${WORK_DIR}/test.jpg"

PYTHON_BIN="python3"
PIP_BIN="pip3"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" 1>&2; }
err() { echo "[ERROR] $*" 1>&2; exit 1; }

ensure_dirs() {
  mkdir -p "${WORK_DIR}" "${MODEL_STORE_DIR}" "${TMP_DIR}"
}

ensure_system_tools() {
  command -v curl >/dev/null 2>&1 || err "curl is required. Please install curl."
  command -v ${PYTHON_BIN} >/dev/null 2>&1 || err "python3 is required. Please install python3."
}

create_venv() {
  if [ ! -d "${VENV_DIR}" ]; then
    log "Creating virtual environment at ${VENV_DIR}"
    ${PYTHON_BIN} -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
}

install_deps() {
  # Keep versions flexible but allow override via env if needed
  local torch_pkg="${TORCH_PKG:-torch}"
  local torchvision_pkg="${TORCHVISION_PKG:-torchvision}"
  local torchserve_pkg="${TORCHSERVE_PKG:-torchserve}"
  local model_archiver_pkg="${MODEL_ARCHIVER_PKG:-torch-model-archiver}"

  log "Installing dependencies: ${torch_pkg} ${torchvision_pkg} ${torchserve_pkg} ${model_archiver_pkg}"
  pip install -q ${torch_pkg} ${torchvision_pkg} ${torchserve_pkg} ${model_archiver_pkg}
}

build_torchscript_model() {
  if [ -f "${SERIALIZED_MODEL_PATH}" ]; then
    log "TorchScript model already exists at ${SERIALIZED_MODEL_PATH}"
    return 0
  fi

  log "Building TorchScript ${MODEL_NAME} model"
  cat > "${TMP_DIR}/export_${MODEL_NAME}.py" << 'PYCODE'
import torch
import torchvision.models as models

def main():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    example_input = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example_input)
    traced.save("MODEL_OUT")

if __name__ == "__main__":
    main()
PYCODE

  python "${TMP_DIR}/export_${MODEL_NAME}.py" || err "Failed to run export script"
  mv MODEL_OUT "${SERIALIZED_MODEL_PATH}"
  log "Saved TorchScript model to ${SERIALIZED_MODEL_PATH}"
}

archive_mar() {
  if [ -f "${MAR_PATH}" ]; then
    log "MAR already exists at ${MAR_PATH}"
    return 0
  fi

  log "Archiving model into MAR at ${MAR_PATH}"
  torch-model-archiver \
    --model-name "${MODEL_NAME}" \
    --version "${MODEL_VERSION}" \
    --serialized-file "${SERIALIZED_MODEL_PATH}" \
    --handler "${HANDLER_NAME}" \
    --export-path "${MODEL_STORE_DIR}"

  log "Created MAR: ${MAR_PATH}"
}

is_port_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn | grep -q ":${port} "
  else
    netstat -ltn | grep -q ":${port} "
  fi
}

start_server() {
  if is_port_listening "${INFERENCE_PORT}"; then
    warn "Port ${INFERENCE_PORT} already in use. TorchServe may already be running."
  fi
  if is_port_listening "${MANAGEMENT_PORT}"; then
    warn "Port ${MANAGEMENT_PORT} already in use. TorchServe management may already be running."
  fi

  log "Starting TorchServe"
  torchserve \
    --start \
    --model-store "${MODEL_STORE_DIR}" \
    --models "${MODEL_NAME}=${MAR_PATH}" \
    --ncs \
    --foreground false \
    --log-config ""

  sleep 2
  wait_for_server
}

wait_for_server() {
  log "Waiting for TorchServe to be ready on port ${INFERENCE_PORT}"
  local retries=60
  local wait_seconds=2
  until curl -s "http://127.0.0.1:${INFERENCE_PORT}/ping" | grep -q '"status": "Healthy"'; do
    retries=$((retries-1))
    if [ ${retries} -le 0 ]; then
      err "TorchServe did not become healthy in time"
    fi
    sleep ${wait_seconds}
  done
  log "TorchServe is healthy"
}

stop_server() {
  if ! is_port_listening "${INFERENCE_PORT}" && ! is_port_listening "${MANAGEMENT_PORT}"; then
    log "TorchServe does not appear to be running"
    return 0
  fi
  log "Stopping TorchServe"
  torchserve --stop || true
  sleep 2
}

download_test_image() {
  local url="${1:-${TEST_IMAGE_URL_DEFAULT}}"
  log "Downloading test image from ${url}"
  curl -fsSL "${url}" -o "${TEST_IMAGE_PATH}"
}

run_test_inference() {
  if [ ! -f "${TEST_IMAGE_PATH}" ]; then
    download_test_image "${TEST_IMAGE_URL_DEFAULT}"
  fi
  log "Running test inference against '${MODEL_NAME}'"
  local response
  response=$(curl -s -X POST "http://127.0.0.1:${INFERENCE_PORT}/predictions/${MODEL_NAME}" -T "${TEST_IMAGE_PATH}") || true
  echo "Inference response:" && echo "${response}"
}

cleanup() {
  log "Stopping server and cleaning up ${WORK_DIR}"
  torchserve --stop || true
  rm -rf "${WORK_DIR}"
}

status() {
  if curl -s "http://127.0.0.1:${INFERENCE_PORT}/ping" | grep -q '"status": "Healthy"'; then
    log "TorchServe status: Healthy"
  else
    warn "TorchServe status: Unavailable"
  fi
  if curl -s "http://127.0.0.1:${MANAGEMENT_PORT}/models" | grep -q "${MODEL_NAME}"; then
    log "Model '${MODEL_NAME}' appears registered"
  else
    warn "Model '${MODEL_NAME}' not listed on management endpoint"
  fi
}

setup() {
  ensure_system_tools
  ensure_dirs
  create_venv
  install_deps
  build_torchscript_model
  archive_mar
}

deploy() {
  setup
  start_server
  run_test_inference
}

print_help() {
  cat <<EOF
TorchServe auto-deploy helper

Commands:
  deploy       Setup -> Start -> Test
  setup        Install deps and build MAR
  start        Start TorchServe with the built model
  test         Run a sample inference request
  stop         Stop TorchServe
  cleanup      Stop server and remove all generated files
  status       Show server and model status

Environment overrides:
  TORCH_PKG, TORCHVISION_PKG, TORCHSERVE_PKG, MODEL_ARCHIVER_PKG
  INFERENCE_PORT, MANAGEMENT_PORT, METRICS_PORT
  MODEL_NAME, MODEL_VERSION
EOF
}

main() {
  local cmd="${1:-deploy}"
  case "${cmd}" in
    deploy)
      setup
      start_server
      run_test_inference
      ;;
    setup)
      setup
      ;;
    start)
      ensure_dirs
      # shellcheck disable=SC1091
      [ -f "${VENV_DIR}/bin/activate" ] && source "${VENV_DIR}/bin/activate" || true
      start_server
      ;;
    test)
      run_test_inference
      ;;
    stop)
      stop_server
      ;;
    cleanup)
      cleanup
      ;;
    status)
      status
      ;;
    help|-h|--help)
      print_help
      ;;
    *)
      warn "Unknown command: ${cmd}"
      print_help
      exit 1
      ;;
  esac
}

main "$@"

