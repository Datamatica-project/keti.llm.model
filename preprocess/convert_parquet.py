from utils.merge import convert_to_dataset, load_datasets
from utils.storage import StorageManager

def main():
    file_path = "C:/Users/dm_ohminchan/Model/data/instrcution"
    data = load_datasets(file_path)
    dataset = convert_to_dataset(data)

    manager = StorageManager(bucket="instruction")
    success = manager.upload_dataset(
        object_name="data/0.0.1v/agriculture.trainset.parquet",
        dataset=dataset
    )
    return success

if __name__ == "__main__":
    print(main())