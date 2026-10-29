from huggingface_hub import snapshot_download
snapshot_download(
        repo_id="pixparse/cc3m-wds",
        repo_type="dataset",
        local_dir="/content/cache/training",
        allow_patterns=[
            "cc3m-tain-0[0-1][0-9][0-9].tar",
	        "cc3m-tain-02[0-4][0-9].tar",
        ]
    )
