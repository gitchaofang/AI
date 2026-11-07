from huggingface_hub import snapshot_download
snapshot_download(
        repo_id="pixparse/cc3m-wds",
        repo_type="dataset",
        local_dir="/content/cache/training",
        allow_patterns=[
            "cc3m-train-0[0-2][0-9][0-9].tar",
            "cc3m-train-0[3-4][5-9][0-9].tar",
            "cc3m-train-05[0-4][0-9].tar",
        ]
    )
