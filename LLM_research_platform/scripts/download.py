from pathlib import Path
import tarfile
import shutil
import time
import json
from google.colab import drive # only run in colab
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download

SHARD_PATH = Path("/content/drive/MyDrive/protected/data/cc3m/shards")
INDEX_PATH = Path("/content/drive/MyDrive/protected/data/cc3m/")
TRAIN_PATH = Path("/content/drive/MyDrive/protected/data/cc3m/training")
VALIDATION_PATH= Path("/content/drive/MyDrive/protected/data/cc3m/validation")


# step 1: download all ".tar" shards
if False:
    snapshot_download(
        repo_id="pixparse/cc3m-wds",
        repo_type="dataset",
        local_dir=SHARD_PATH,
        allow_patterns=[
            "cc3m-train-*.tar",
            "cc3m-validation-*.tar",
        ]
    )

# step 2: build index dict and seperate ".tar" files in training and validation files:
cnt = 0
index_dict = {} 
'''
index_dict:
    key: file_name
    value: file address
'''
if True:
    for file_path in SHARD_PATH.iterdir():
        cnt += 1
        if(cnt >= 10):
            break 
        file_name = file_path.name
        if file_path.is_file():
            if file_path.suffix != ".tar":
                continue
            print(f"{file_name}")

            # iterate over files inside .tar and update index_dict
            with tarfile.open(file_path, "r") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    index_dict[member.name].split('.')[0] = file_name

            # copy **-train-**.tar to training dir and **-validation-**.tar to validation dir
            type = file_name.split('.')[0].split('-')[1]
            if type == "train":
                shutil.copy2(file_path, TRAIN_PATH)
                print(f"{file_name} is copied to {TRAIN_PATH}")
            elif type == "validation":
                shutil.copy2(file_path, VALIDATION_PATH)
                print(f"{file_name} is copied to {VALIDATION_PATH}")


#    for key, value in index_dict.items():
#        print(f"{key}: {value}")
    index_dict_path = INDEX_PATH/"index.json"
    with open(index_dict_path,"w") as f:
        json.dump(index_dict, f, indent=2)  
    print(f"Saved {len(index_dict)} entries to {INDEX_PATH}")   
            