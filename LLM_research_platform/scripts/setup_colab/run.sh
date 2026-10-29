mkdir ./cache
mkdir ./cache/training
mkdir ./cach/validation
mkdir ./token_train
cd ././token_train
curl -L -o ./wikitext-103-raw-v1.zip https://wikitext.smerity.com/wikitext-103-raw-v1.zip
unzip -q ./wikitext-103-raw-v1.zip -d ./
cp ./wikitext-103-raw/wiki.train.raw train.txt
cd ./content/cache/validation
python3 download_validation_samples.py
for f in cc3m-validation-0[0-9][0-9][0-9].tar; do
    echo "Extracting $f"
    tar -xf "$f" && rm "$f" || break
done
cd ./content/cache/training
python3 download_training_samples.py
for f in cc3m--0[0-9][0-9][0-9].tar; do
    echo "Extracting $f"
    tar -xf "$f" && rm "$f" || break
done
python3 pick_patches.py

