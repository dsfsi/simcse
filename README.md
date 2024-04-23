
## Project Description 
This repository contains the trained model for our paper (under review) that is based on the original implementation of SimCSE [1]. The original model was adapted for DNA downstream tasks by training it on
a small sample size k-mer tokens generated from the human reference genome, and can be used to generate sentence embeddings for DNA tasks.

###  Prerequisites 
-----------
Please see the original [SimCSE](https://github.com/princeton-nlp/SimCSE) for installation details. We only share the fine-tuned model as well as our custom tokenizer, which can be found here.

### Usage 

Download the model in to a directory then run the following code to get the sentence embeddings:

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
The retrieved embeddings can be utilized as input for a machine learning classifier to perform classification or similarity tasks.

## Authors 
-----------

* Written by : 
* Contact details : u19394277@tuks.co.za

### References

<a id="1">[1]</a> 
Gao, Tianyu, Xingcheng Yao, and Danqi Chen. "Simcse: Simple contrastive learning of sentence embeddings." arXiv preprint arXiv:2104.08821 (2021).


## More Information 
---------

We're working on an upgraded model with a whole new architecture, and it will be released soon.
