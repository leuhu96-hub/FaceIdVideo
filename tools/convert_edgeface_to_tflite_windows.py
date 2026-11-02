"""
Windows fallback for convert_edgeface_to_tflite.py (ai-edge-torch has no Windows build):
torch -> ONNX -> onnx2tf -> TFLite. Same output: float32, NHWC input [1,112,112,3], no Flex ops.

    pip install torch timm tensorflow==2.20.* onnx onnxscript onnxsim onnx2tf onnx_graphsurgeon \
                sng4onnx psutil tf_keras ai-edge-litert
    python convert_edgeface_to_tflite_windows.py --out ../app/src/main/assets/face_embedder.tflite

Use a short venv path (e.g. C:\\Users\\<you>\\fv_venv): TensorFlow hits the Windows MAX_PATH limit.
If gitlab.idiap.ch is unreachable, put the weights in ~/.cache/torch/hub/checkpoints/ first, e.g.
https://huggingface.co/Idiap/EdgeFace-XS-GAMMA/resolve/main/edgeface_xs_gamma_06.pt
"""
import argparse, os, shutil, sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="edgeface_xs_gamma_06")
ap.add_argument("--out", default="face_embedder.tflite")
ap.add_argument("--work", default="edgeface_work")
args = ap.parse_args()
os.makedirs(args.work, exist_ok=True)

model = torch.hub.load("otroshi/edgeface", args.variant, source="github", pretrained=True, trust_repo=True).eval()
x = torch.randn(1, 3, 112, 112)
with torch.no_grad():
    ref = model(x).numpy()

# EdgeNeXt's Fourier positional encoding only depends on the (fixed) input shape:
# bake it as a constant buffer so the converters don't have to handle cumsum/sin/cos.
from timm.models.edgenext import PositionalEncodingFourier

pe_modules = [m for m in model.modules() if isinstance(m, PositionalEncodingFourier)]
captured = {}
hooks = [m.register_forward_hook(lambda mod, inp, out: captured.__setitem__(mod, out.detach().clone()))
         for m in pe_modules]
with torch.no_grad():
    model(x)
for h in hooks:
    h.remove()
for m in pe_modules:
    m.register_buffer("baked", captured[m])
    m.forward = (lambda mod: (lambda shape: mod.baked))(m)
print(f"baked {len(pe_modules)} positional encodings")

# x.chunk() exports as an ONNX Split that onnx2tf mis-shapes; use static slices instead (same math).
import math
from timm.models.edgenext import SplitTransposeBlock

_orig_chunk = torch.Tensor.chunk


def _static_chunk(t, chunks, dim=0):
    if dim != 1:
        return _orig_chunk(t, chunks, dim)
    c = t.shape[1]
    step = math.ceil(c / chunks)
    return tuple(t[:, i:min(i + step, c)] for i in range(0, c, step))


_orig_stb_forward = SplitTransposeBlock.forward


_orig_unbind = torch.Tensor.unbind


def _static_unbind(t, dim=0):
    return tuple(t.select(dim, i) for i in range(t.shape[dim]))


def _stb_forward(self, x):
    torch.Tensor.chunk = _static_chunk
    torch.Tensor.unbind = _static_unbind  # XCA: q, k, v = qkv.unbind(0)
    try:
        return _orig_stb_forward(self, x)
    finally:
        torch.Tensor.chunk = _orig_chunk
        torch.Tensor.unbind = _orig_unbind


SplitTransposeBlock.forward = _stb_forward
with torch.no_grad():
    assert np.allclose(model(x).numpy(), ref, atol=1e-5)

onnx_path = os.path.join(args.work, "edgeface.onnx")
torch.onnx.export(model, (x,), onnx_path, input_names=["input"], output_names=["embedding"],
                  opset_version=17, dynamo=False)

import onnx2tf
tf_dir = os.path.join(args.work, "tf")
onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=tf_dir,
                not_use_onnxsim=False, verbosity="error", tflite_backend="tf_converter",
                replace_to_pseudo_operators=["Erf"])
tfl = os.path.join(tf_dir, "edgeface_float32.tflite")

from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path=tfl)
it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]
print("tflite input", inp["shape"], inp["dtype"], "output", out["shape"])
it.set_tensor(inp["index"], x.permute(0, 2, 3, 1).numpy().astype(np.float32))
it.invoke()
got = it.get_tensor(out["index"])
cos = float((got.ravel() @ ref.ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref)))
print(f"embedding dim = {ref.shape[-1]}, max |tflite - torch| = {np.abs(got - ref).max():.2e}, cos = {cos:.6f}")
if cos < 0.999:
    sys.exit("conversion mismatch")
shutil.copy(tfl, args.out)
print("saved", args.out)
