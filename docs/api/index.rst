=============
API Reference
=============

The API reference gives an overview of `TopoBenchX`, which consists of several modules:

- `data` implements the utilities to download  load, and preprocess data, among other functionalities.
- `dataloader` implements custom dataloaders to generate batches from topological data.
- `evaluator` implements functionalities to evaluate the performances of the neural networks.
- `loss` implements the loss functions.
- `model` implements the classes to handle the neural networks.
- `nn` implements utilities regarding neural networks.
- `optimizer` implements funtionalities to manage both optimizers and schedulers.
- `transforms` implements utilities to preprocess datasets, including lifting procedures.
- `utils` implements utilities to handle the training process.

.. toctree::
   :maxdepth: 2
   :caption: Packages & Modules

   data/index
   dataloader/index
   evaluator/index
   loss/index
   model/index
   nn/index
   optimizer/index
   transforms/index
   utils/index
