#Import Libraries
import pandas as pd
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
file_path = project_root/"data"/"raw"/"BHP.csv"

print("rusnning from:", current_file)
print("Looking for data at:", file_path)
print("File exists:", file_path.exists())

df = pd.read_csv(file_path)

print(df.head())
print(df.shape)