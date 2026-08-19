import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import get_model


def test_model_output_shape():
    model = get_model("cnn", num_classes=10)
    model.eval()
    batch = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (2, 10)