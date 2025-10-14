import pandas as pd
from scripts.preprocessing import preprocess
import os

def test_preprocess(tmp_path):
    # Create dummy data
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)
    preprocess(str(input_path), str(output_path))
    df_out = pd.read_csv(output_path)
    assert df_out.shape == df.shape
    assert abs(df_out["A"].mean()) < 1e-6
    assert abs(df_out["B"].mean()) < 1e-6
