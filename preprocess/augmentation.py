from utils.storage import StorageManager
from utils.generate import run_minio_qa_pipeline

def main():
    storage = StorageManager("chunk")
    result = run_minio_qa_pipeline(storage, prefix="data/",group_size=3,output_file="C:/Users/dm_ohminchan/Model/data/instrcution/qa_set.json")
    return result

if __name__ == "__main__":
    main()