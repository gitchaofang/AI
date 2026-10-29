from huggingface_hub import snapshot_download
snapshot_download(
        repo_id="pixparse/cc3m-wds",
        repo_type="dataset",
        local_dir="/content/cache/validation",
        allow_patterns=[
            "cc3m-validation-0[0-9][0-9][0-9].tar",
        ]
    )
