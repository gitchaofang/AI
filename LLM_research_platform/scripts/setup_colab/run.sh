# 1. Create the necessary directory structure properly
mkdir -p /cache/training
mkdir -p /cache/validation
mkdir -p /token_train

# 2. Download and extract the WikiText dataset
cd /content/token_train
curl -L -o ./wikitext-103-raw-v1.zip https://wikitext.smerity.com/wikitext-103-raw-v1.zip
unzip -q ./wikitext-103-raw-v1.zip -d ./
cp ./wikitext-103-raw/wiki.train.raw train.txt

# 3. Navigate back out to the validation cache and run scripts
# (Using absolute paths ensures you don't get lost in relative directories)
cd /content/cache/validation
python3 download_validation_samples.py
for f in cc3m-validation-0[0-9][0-9][0-9].tar; do
    echo "Extracting $f"
    tar -xf "$f" && rm "$f" || break
done

# 4. Navigate to the training cache and process
cd /content/cache/training
python3 download_training_samples.py
for f in cc3m-0[0-9][0-9][0-9].tar; do
    echo "Extracting $f"
    tar -xf "$f" && rm "$f" || break
done

# 5. Run the final processing script
cd /content/AI/LLM_research_platform/scripts/setup_colab
python3 pick_patches.py

