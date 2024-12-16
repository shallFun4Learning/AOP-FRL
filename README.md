# AOP-FRL: A Novel Method for Antioxidant Peptides Identification Based on Multi-Feature Representation Learning

## Overview

The **AOP-FRL** algorithm is designed to identify antioxidant peptides using a combination of sequnce and structral features. 
This repository contains the code for training, inference, and dataset preprocessing for this novel method.

We provide local source code and Notebooks that can be run online on Google Colab. 
If you only want to learn about the information for online notebooks, you can click [OnlineNotebooks](#OnlineNotebooks).


## Features

- **Graph Convolutional Network (GCN)** for molecular structure feature extraction.
- **Protein Language Model** for sequence-based feature extraction from peptide sequences.
- **Multi-Feature Fusion**: Combines both sequence and structure information for improved prediction accuracy.
- **Cross-Validation**: Implements 5-fold cross-validation for model evaluation.
- **Evaluation Metrics**: Supports multiple metrics such as accuracy, Matthews correlation coefficient (MCC), AUC, and recall.

## Installation

Ensure that you have Python 3.8.3 installed. You will also need to install the following dependencies:

```bash
pip install torch==1.12.0
pip install transformers==4.32.1 
pip install Accelerate==0.24.1
pip install datasets==2.16.1
pip install rdkit==2024.3.2
pip install torch-geometric==2.3.1
pip install pandas
pip install sklearn
```
Please check the version number carefully, which is our best practice environment. The mismatched version may cause unknown results.
If using GPU, make sure you have the necessary CUDA drivers installed for PyTorch to utilize GPU acceleration.

## Usage
### Preparing the Dataset
The original data set is published in [AnOxPePred](https://github.com/TobiasHeOl/AnOxPePred). 

You can also select the data set we have processed, which can be found here [Onedirve-datasets](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/ERbG9MF98UBAuZ-oPJATAz0Btb3CofQPlCIIjSzK_t8mFw). Then you can skip the data processing step.

Before training or inference, you need a CSV file with peptide sequences and associated labels. Each sequence should be in the sequence column, and labels should be in the label column. Additionally, the SMILES representations for the peptides are also used and can be generated by the script getSMILES.py.

Generating SMILES Representations
To generate SMILES representations for your peptides, run the following script:
```bash
python getSMILES.py
```
This will process the peptide sequences, convert them into SMILES notation, and generate a new CSV with the added SMILES column.

## Training the Model
To train the model, you need to modify the file paths in AOP_FRL_train.py. Specifically, you need to modify the following parameters:

- Data File Path (data_file_path): Set this to the path of your CSV file. The file should contain sequences(and SMILES) and their corresponding labels.

- Model Output Path (out_file_path): Set this to the directory where you want to save the trained model and results.

Modify the following part in the code:

You need to download ESM-2 which could be found here [HuggingFace](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main).

```python
data_file_path = "/path/to/your/data.csv"  # Replace with the path to your data file
esm2_model_path = '/path/to/esm2_model'    # Replace with the path to your ESM-2 model
out_file_path = '/path/to/output_dir'      # Replace with the directory to save the model and results
```
Then, run the following command to start training:
```bash
python AOP_FRL_train.py
```

##  Inference
To make predictions with a trained model, you also need to modify the following paths in AOP_FRL_infer.py:
- Data File Path (data_file_path): Set this to the path of your test data CSV file.
- Model File Path (model_file_path): Set this to the path of the trained model, which is typically the checkpoint directory saved during training.
- Results Path (res_path): Set this to the directory where you want to save the inference results.

Our trained model weights file could be found here [Onedrive-Weights](https://4wgz12-my.sharepoint.com/:f:/g/personal/admin_4wgz12_onmicrosoft_com/Esy88sjlqn5KldBd22tmvx4BM0y0VfqhkdTWJINbtSEodg?e=DHy3Lm).

Modify the following part in the code:
```python
data_file_path = "/path/to/your/test_data.csv"  # Path to your test data file
model_file_path = '/path/to/your/trained_model'  # Path to the trained model
res_path = '/path/to/results_output'  # Directory to save the inference results
```
Then, run the following command to perform inference:
```bash
python AOP_FRL_infer.py
```
Inference results will be saved in the specified directory, including predicted labels and scores.

## Evaluation Metrics
The model's performance will automatically be evaluated using the following metrics during training and inference:

- Accuracy (ACC): The proportion of correct predictions.
- Matthews Correlation Coefficient (MCC): A balanced measure of classification performance.
- Recall: True Positive Rate (Sensitivity).
- AUC (Area Under Curve): The area under the ROC curve, measuring the model’s ability to distinguish between classes.

These metrics will be calculated and saved using the Getmetrics.py module after training and inference.

## OnlineNotebooks
We have prepared notebooks that can be run online on Google Colab. The usage instructions are as follows:

1. Download the source code [Onedrive-Source](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/EQXYmjaXJylEhcp05pKEwXkBuAFyl5nikhb6wDdw5y9IOg?e=zrabg2) package we have prepared and upload it to your Google Drive.

2. Download the dataset provided by [AnOxPePred](https://github.com/TobiasHeOl/AnOxPePred), use rdkit to derive SIMLES, retain the FRS and partition columns, and rename FRS to label. The final column names should include [sequence, label, partition, SMILES]. Alternatively, you can use the preprocessed version we provided (available for download here: [Onedirve-datasets](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/ERbG9MF98UBAuZ-oPJATAz0Btb3CofQPlCIIjSzK_t8mFw)). Copy the dataset to the data folder.

3. Download the ESM-2 150M model file (available here: [HuggingFace](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main)), and copy it to the corresponding folder.

4. Use our notebooks([Colab](https://colab.research.google.com/drive/1lo31jFqFnlDDrgxHaQHIp8QjY9dY6ohb?usp=sharing)). You need to modify the paths in the code to align with the paths in your Google Drive.

5. You have completed all the steps. Now, run it. ^_^


## License
Academic Free License (AFL) v. 3.0

## Supports
Feel free to submit an issue or contact the author(sfun@foxmail.com) if you encounter any problems during use.

Happy New Year 2025 (is coming~)

:-)
