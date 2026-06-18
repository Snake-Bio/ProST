# ProST: an image prompt-guided multimodal representation learning framework for spatial domain identification


## Requirements

You'll need to install the following packages in order to run the codes.

Python==3.10.19
numpy==1.26.4
pandas==2.0.3
scipy==1.10.1
stlearn==0.4.8
pytorch==2.1.2+cu121
torch_geometric==2.7.0


## Usage

### Raw Data Preparation

Place the raw spatial transcriptomics data (e.g., DLPFC) in the folder ***Data***.


### Model Training and Testing

Run ***ProST/DLPFC.py*** to train and test the ProST model:

`python DLPFC.py`


