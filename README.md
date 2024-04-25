
## Project Description 
This repository contains the trained model for our paper: **Fine-tuning a Sentence Transformer for DNA & Protein tasks** that is currently under review at BMC Bioinformatics. This model, called **simcse-dna**; is based on the original implementation of **SimCSE [1]**. The original model was adapted for DNA downstream tasks by training it on a small sample size k-mer tokens generated from the human reference genome, and can be used to generate sentence embeddings for DNA tasks.

###  Prerequisites 
-----------
Please see the original [SimCSE](https://github.com/princeton-nlp/SimCSE) for installation details. The model will be hosted on Zenodo (DOI: 10.5281/zenodo.11046580). It 
is also available on 🤗 [huggingface](https://huggingface.co/dsfsi/simcse-dna).

### Usage 

Download the model into a directory or 🤗 [huggingface](https://huggingface.co/dsfsi/simcse-dna) then run the following code to get the sentence embeddings:

```python 

import torch
from transformers import AutoModel, AutoTokenizer

# Import trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/model/directory/")
model = AutoModel.from_pretrained("/path/to/model/directory/")


#sentences is your list of n DNA tokens of size 6 
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output


```
The retrieved embeddings can be utilized as input for a machine learning classifier to perform classification.

## Performance on evaluation tasks

Find out more about the datasets and access in the paper **(TBA)**

### Task 1: Detection of colorectal cancer cases (after oversampling)

|  | 5-fold Cross Validation accuracy | Test accuracy |
| --- | --- | ---|
| LightGBM | 91 | 63 |
| Random Forest | **94** | **71** |
| XGBoost | 93 | 66 |
| CNN | 42 | 52 |

| | 5-fold Cross Validation F1 | Test F1 |
| --- | --- | ---|
| LightGBM |  91 | 66 |
| Random Forest |  **94** | **72** |
| XGBoost | 93 | 66 |
| CNN |  41 | 60 |

### Task 2: Prediction of the Gleason grade group (after oversampling)

|  | 5-fold Cross Validation accuracy | Test accuracy |
| --- | --- | ---|
| LightGBM | 97 | 68 |
| Random Forest | **98** | **78** |
| XGBoost |97 | 70 |
| CNN |  35 |  50 |

| | 5-fold Cross Validation F1 | Test F1 |
| --- | --- | ---|
| LightGBM |  97 |  70 |
| Random Forest | **98** | **80** |
| XGBoost |97 | 70 |
| CNN |  33 | 59 |

### Task 3: Detection of human TATA sequences (after oversampling)

|  | 5-fold Cross Validation accuracy | Test accuracy |
| --- | --- | ---|
| LightGBM | 98  | 93  |
| Random Forest | **99** | **96** |
| XGBoost |**99** | 95 |
| CNN | 38  | 59 |

| | 5-fold Cross Validation F1 | Test F1 |
| --- | --- | ---|
| LightGBM | 98 | 92 |
| Random Forest | **99** | **95** |
| XGBoost | **99** | 92 |
| CNN |  58 | 10 |


## Authors 
-----------

* Written by : Mpho Mokoatle, Vukosi Marivate, Darlington Mapiye, Riana Bornman, Vanessa M. Hayes
* Contact details : u19394277@tuks.co.za

## Citation 
-----------
Bibtex Reference **TBA**


## License
* cc-by-sa-4.0

### References

<a id="1">[1]</a> 
Gao, Tianyu, Xingcheng Yao, and Danqi Chen. "Simcse: Simple contrastive learning of sentence embeddings." arXiv preprint arXiv:2104.08821 (2021).


## More Information 
---------

We're working on an upgraded model with a whole new architecture, and it will be released soon.
