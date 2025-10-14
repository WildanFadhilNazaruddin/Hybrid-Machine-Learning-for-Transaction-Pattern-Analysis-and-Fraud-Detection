import pandas as pd
from scripts.clustering import cluster
import os

def test_cluster(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 5, 6, 7]})
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)
    cluster(str(input_path), str(output_path), n_clusters=2)
    df_out = pd.read_csv(output_path)
    assert "cluster" in df_out.columns
    assert df_out["cluster"].nunique() == 2
