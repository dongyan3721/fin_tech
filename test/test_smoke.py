r"""冒烟测试：验证数据就位、模型可实例化并完成一次前向。
运行：  .\.venv\Scripts\python.exe -m pytest test  （或直接 python test/test_smoke.py）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data" / "processed"


def test_data_files_exist():
    for name in ["financial_indicators_robust.csv",
                 "kmv_analysis_results.csv",
                 "combined_edges.xlsx"]:
        assert (DATA_DIR / name).exists(), f"缺少数据文件: {name}"


def test_model_forward():
    import torch
    from legacy.train_tgc import TGCN

    model = TGCN(input_dim=10, hidden_dim=64, seq_len=3)
    x = torch.randn(8, 3, 10)  # (batch, seq_len, features)
    out = model(x)
    assert out.shape == (8, 1)
    assert torch.all((out >= 0) & (out <= 1)), "输出应为 sigmoid 概率"


if __name__ == "__main__":
    test_data_files_exist()
    test_model_forward()
    print("smoke test passed.")
