# 1.Download files from Google Drive
The following directories are not included in the repository due to size:
- `datasets/`
- `encoded_models_new/`
- `finetuned_models/`
- `keys/`
You can download them separately from the following Google Drive link:  
https://drive.google.com/drive/folders/1Uzsvpmw_IoZy9mC6Z-PUNfgN77LG1Oxc?usp=sharing  
Put them directly inside your testing directory, as a whole folder just as you downloaded.

# 2. Download Github files and Install required modules
2-1. git clone https://github.com/crypto-starlab/THOR_demo.git  
2-2. pip install requirements.txt  
2-3. Install Desilo Library
- python setup.py Install
- pip install -e .
- Copy files in resource in Google Drive to liberate/src/liberate/fhe/cache/resources

# 3. Run HE Model!
Run with forward.ipynb 

Note : Python 3.10 required since the bootstrapping code is protected with PyArmor
