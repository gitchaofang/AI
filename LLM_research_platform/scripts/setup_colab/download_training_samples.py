from huggingface_hub import snapshot_download
snapshot_download(
        repo_id="pixparse/cc3m-wds",
        repo_type="dataset",
        local_dir="/content/cache/training",
        allow_patterns=[
            "cc3m-train-0000.tar",
        #    "cc3m-train-0[0-3][0-5][0-9].tar",
        #    "cc3m-train-0[4-5][6-9][0-9].tar",
        ]
    )
