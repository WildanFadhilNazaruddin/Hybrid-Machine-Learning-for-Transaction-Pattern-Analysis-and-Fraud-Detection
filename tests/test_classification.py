import pandas as pd
from scripts.classification import classify
import os

def test_classify(capsys, tmp_path):
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [4, 5, 6, 7],
        "cluster": [0, 1, 0, 1]
    })
    input_path = tmp_path / "input.csv"
    df.to_csv(input_path, index=False)
    classify(str(input_path))
    captured = capsys.readouterr()
    assert "precision" in captured.out
