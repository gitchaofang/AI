from pathlib import Path
import tarfile
import shutil
import json
from huggingface_hub import snapshot_download

INDEX_PATH = Path("/content/drive/MyDrive/protected/data/cc3m/")
INDEX_PATH.mkdir(parents=True, exist_ok=True)
SHARD_PATH = Path("/content/drive/MyDrive/protected/data/cc3m/shards")
SHARD_PATH.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = Path("/content/drive/MyDrive/protected/data/cc3m/training")
TRAIN_PATH.mkdir(parents=True, exist_ok=True)
VALIDATION_PATH= Path("/content/drive/MyDrive/protected/data/cc3m/validation")
VALIDATION_PATH.mkdir(parents=True, exist_ok=True)
PATCH_SIZE = 16

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

'''
index_dict:
    key: file_name
    value: file address
'''
index_dict = {}
index_dict_train = {}
index_dict_valid = {}

'''
metadata:
max_len_patch: longest patchified sequence 
max_len_text: longggest tokenized text sequence
'''
max_len_patch = 0
max_len_text = 0

# iterate shards and seperate them
if True:
    for file_path in SHARD_PATH.iterdir():
        file_name = file_path.name
        shard_type = file_name.split('.')[0].split('-')[1]
        if file_path.is_file():
            if file_path.suffix != ".tar":
                continue
            print(f"{file_name}")

            # iterate over files inside .tar and update index_dict
            with tarfile.open(file_path, "r") as tar:
                for member in tar:
                    if not member.isfile():
                        continue

                    # for metadata
                    if member.name.endswith(".json"):
                        # update index dictionaries
                        sample_id = Path(member.name).stem
                        index_dict[sample_id] = file_name

                        if shard_type == "train":
                            index_dict_train[sample_id] = file_name
                        elif shard_type == "validation":
                            index_dict_valid[sample_id] = file_name

                        cnt += 1
                        with tar.extractfile(member) as f:
                            data = json.load(f)
                            width = int(data["width"])
                            height = int(data["height"])
                            caption = str(data["caption"])
                            patch_seq_len = (width//PATCH_SIZE) * (height // PATCH_SIZE)
                            text_len = (len(caption) + 4) // 4
                            max_len_patch = max(max_len_patch,patch_seq_len)
                            max_len_text = max(max_len_text,text_len)


            if False:
                # copy **-train-**.tar to training dir and **-validation-**.tar to validation dir 
                if shard_type == "train":
                    shutil.copy2(file_path, TRAIN_PATH)
                    print(f"{file_name} is copied to {TRAIN_PATH}")
                elif shard_type == "validation":
                    shutil.copy2(file_path, VALIDATION_PATH)
                    print(f"{file_name} is copied to {VALIDATION_PATH}")


#    for key, value in index_dict.items():
#        print(f"{key}: {value}")
    if True:
        # store over all index
        index_dict_path = INDEX_PATH/"index.json"
        with open(index_dict_path,"w") as f:
            json.dump(index_dict, f, indent=2)  
        print(f"Saved {len(index_dict)} entries to {INDEX_PATH}")

        # store index for training data
        index_dict_train_path = TRAIN_PATH/"index.json"
        with open(index_dict_train_path,"w") as f:
            json.dump(index_dict_train, f, indent=2)  
        print(f"Saved {len(index_dict_train)} entries to {TRAIN_PATH}")

        # store index for validation
        index_dict_valid_path = VALIDATION_PATH/"index.json"
        with open(index_dict_valid_path,"w") as f:
            json.dump(index_dict_valid, f, indent=2)  
        print(f"Saved {len(index_dict_valid)} entries to {VALIDATION_PATH}")  
    # Store metadata:
    if True:
        meta_dict = {
            "total samples": cnt,
            "max_len_patch": max_len_patch,
            "max_len_text": max_len_text,
        }
        meta_dict_path = INDEX_PATH/"metadata.json"
        with open(meta_dict_path, "w") as f:
            json.dump( meta_dict,f,indent=2)
        print(f"Saved metadata to {INDEX_PATH}")  


            