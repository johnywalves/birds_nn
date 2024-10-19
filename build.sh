rm -rf data
rm -rf grayscale
rm -rf model

python3 01_prepare.py
python3 02_separate.py
python3 03_classification.py
