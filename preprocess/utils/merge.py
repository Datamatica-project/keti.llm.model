import glob
import json
import pandas as pd

def load_datasets(json_dir: str, pattern: str = "*.json"):
    data = []

    for file_path in glob.glob(f"{json_dir}/{pattern}"):
        with open(file_path, "r", encoding="utf-8") as f:
            data.extend(json.load(f))
    return data

def convert_to_dataset(data: list):
    return pd.DataFrame.from_records(data)




