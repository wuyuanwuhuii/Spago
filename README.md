# Spago
The development of spatial multi-omics data can provide more biologically meaningful explanations for studying disease mechanisms and implementing precision medicine applications. To address the current lack of spatial multi-omics data, we have designed a deep generative model based on VGAE, named Spago. Through the trained model, Spago can accurately and efficiently generate between multi-omics data.

## Framework
![framework](./algorithm.png)

## Installation

To install Spago, make sure you have [PyTorch](https://pytorch.org/) and [scanpy](https://scanpy.readthedocs.io/en/stable/) installed. If you need more details on the dependences, look at the `environment.yml` file. 

### Train model
With the datasets obtained after pre-processing, we can then train the model:

```bash
python bin/train_predict_rna.py --outdir output
```
### Generation on other datasets
Once trained, Spago can generate paired datasets from other datasets using the following example command.

```bash
python bin/predict-rna.py --outdir Otherdataset_generation 
python bin/predict-atac.py --outdir Otherdataset_generation 
```

### Additional commandline options
All the above scripts have some options designed for advanced users, exposing some features such as clustering methods, learning rates, etc. Users can adjust them by themselves.
