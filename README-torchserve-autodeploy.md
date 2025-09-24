## TorchServe Auto-Deploy Script

This repository includes a helper script to automatically set up TorchServe, package a public pretrained `resnet18` model, start the server, and run a sample inference.

### Prerequisites
- Linux with `bash`, `python3`, and `curl` installed.
- Internet connectivity.

### Quickstart
```bash
/workspace/scripts/torchserve_autodeploy.sh deploy
```

This will:
- Create a Python virtual environment under `/workspace/torchserve_autodeploy/.venv`
- Install `torch`, `torchvision`, `torchserve`, and `torch-model-archiver`
- Export TorchScript for pretrained ResNet-18
- Archive to a `.mar` and place it in the model store
- Start TorchServe and run an inference on a sample image

### Commands
- `deploy`: Full flow (setup → start → test)
- `setup`: Install dependencies and build `.mar`
- `start`: Start TorchServe with the packaged model
- `test`: Run a sample prediction
- `stop`: Stop TorchServe
- `cleanup`: Stop and remove generated files
- `status`: Print server and model status

### Environment Overrides
You can override package pins or model/server parameters via environment variables:
- `TORCH_PKG`, `TORCHVISION_PKG`, `TORCHSERVE_PKG`, `MODEL_ARCHIVER_PKG`
- `INFERENCE_PORT`, `MANAGEMENT_PORT`, `METRICS_PORT`
- `MODEL_NAME`, `MODEL_VERSION`

Example:
```bash
TORCH_PKG=torch==2.3.1 TORCHVISION_PKG=torchvision==0.18.1 \
/workspace/scripts/torchserve_autodeploy.sh deploy
```

### Outputs
- Model store: `/workspace/torchserve_autodeploy/model_store`
- TorchScript file: `/workspace/torchserve_autodeploy/resnet18_scripted.pt`
- Test image: `/workspace/torchserve_autodeploy/test.jpg`

### Notes
- The default handler is `image_classifier`. For custom handlers, replace the `--handler` in the script.
- If ports 8080/8081 are occupied, stop the existing service or override ports via env.

