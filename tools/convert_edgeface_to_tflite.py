"""
Convert EdgeFace (PyTorch) -> face_embedder.tflite (float32, NHWC input [1,112,112,3], RGB,
normalized (x - 127.5) / 127.5) for app/src/main/assets/.

    pip install torch ai-edge-torch timm
    python convert_edgeface_to_tflite.py --variant edgeface_xs_gamma_06

Variants: edgeface_xs_gamma_06 (1.77M params, fastest), edgeface_s_gamma_05, edgeface_base.
Check the license of the pretrained weights before shipping a commercial app.
"""
import argparse

import torch
import ai_edge_torch


class NHWCWrapper(torch.nn.Module):
    """The app feeds NHWC tensors; EdgeFace expects NCHW."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.permute(0, 3, 1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="edgeface_xs_gamma_06")
    ap.add_argument("--out", default="face_embedder.tflite")
    args = ap.parse_args()

    model = torch.hub.load("otroshi/edgeface", args.variant, source="github", pretrained=True)
    model = NHWCWrapper(model).eval()
    sample = (torch.randn(1, 112, 112, 3),)

    with torch.no_grad():
        ref = model(*sample)
    edge = ai_edge_torch.convert(model, sample)
    out = edge(*sample)
    diff = float(abs(torch.tensor(out) - ref).max())
    print(f"embedding dim = {ref.shape[-1]}, max |tflite - torch| = {diff:.2e}")
    edge.export(args.out)
    print("saved", args.out, "-> copy to app/src/main/assets/")


if __name__ == "__main__":
    main()
