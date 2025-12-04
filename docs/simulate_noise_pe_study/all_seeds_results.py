
import glob
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

pattern = str(BASE_DIR /"outdir_simulate_pe_study"/ "seed_*" / "seed_*_results.csv")
csv_files = sorted(glob.glob(pattern))

dfs = [pd.read_csv(f) for f in csv_files]
big_df = pd.concat(dfs, ignore_index=True)

out_path = BASE_DIR / "all_seeds_results.csv"
big_df.to_csv(out_path, index=False)
print("Saved merged csv to:", out_path)
