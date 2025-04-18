PART-A
📄 cnn_model.py
This file contains the implementation of a configurable Convolutional Neural Network (CNN) using PyTorch Lightning. The model is designed to support hyperparameter tuning and experimentation with different architectures. Key features include:

Customizable convolutional layers: You can specify the number of filters, kernel sizes, and whether to use batch normalization and dropout.

Flexible activation functions: Supports ReLU, GELU, SiLU, and Mish activations.

Automatic dimension handling: The model dynamically computes the size of the flattened feature vector using a dummy input, which simplifies architecture changes.

Fully connected layers: After the convolutional blocks, the model uses a dense layer before the final classification layer.

Training, validation, and test steps: Implemented using PyTorch Lightning’s modular structure with built-in logging for loss and accuracy.

Optimizer: Uses the Adam optimizer with a configurable learning rate.

This modular design allows easy integration with hyperparameter sweep tools like Weights & Biases (W&B) for automated model tuning.

📄 dataset.py (Nature12KDataModule)
This file defines a custom Nature12KDataModule using PyTorch Lightning’s LightningDataModule. It handles data preparation, transformation, and loading for the iNaturalist_12K dataset. Key features include:

Automatic download and extraction: If the dataset is not found locally, it is automatically downloaded and extracted.

Support for data augmentation: When enabled, it applies transformations such as random flips, rotations, and color jittering to improve generalization.

Image preprocessing: All images are resized to a specified shape, normalized using ImageNet statistics, and converted to tensors.

Train-validation split: The training data is split into training and validation sets (80/20 split).

Flexible batch size and image size: These parameters can be customized when initializing the module.

Dataloaders: Provides separate train_dataloader, val_dataloader, and test_dataloader for easy integration with PyTorch Lightning models.

This module simplifies and standardizes data handling for training and evaluating CNN models on the iNaturalist_12K dataset.


🔁 sweep_train.py (W&B Hyperparameter Sweep Script)
This script performs automated hyperparameter tuning using Weights & Biases (W&B) Sweeps for training a CNN on the iNaturalist_12K dataset. It integrates PyTorch Lightning for model training and evaluation.

🧠 What It Does:
Loads the CNN model and Nature12KDataModule for handling the dataset.

Defines a train() function that:

Initializes the W&B run and reads the current sweep configuration.

Loads and prepares the dataset with optional data augmentation.

Instantiates the CNN model with hyperparameters from the sweep config.

Trains the model using PyTorch Lightning’s Trainer.

Evaluates the model on the test set.

Sets up a W&B sweep configuration using the Bayesian optimization strategy to maximize validation accuracy.

Tunes the following hyperparameters:

conv_filters: Number of filters in each convolutional layer.

kernel_sizes: Size of the convolution kernels.

activation: Activation function (ReLU, GELU, SiLU, Mish).

dense_neurons: Number of neurons in the dense layer.

lr: Learning rate.

batch_norm: Whether to use batch normalization.

dropout: Dropout rate after each layer.

batch_size: Batch size for training.

data_augmentation: Whether to apply data augmentation.

🚀 How it works:
The sweep is launched using:

wandb.agent(sweep_id, function=train, count=100)

which runs the train() function 100 times with different hyperparameter combinations to find the best-performing configuration.

This script enables scalable and efficient tuning of CNN architectures using W&B, helping identify the most effective model setup for the iNaturalist_12K dataset.

✅ best_model_train.py (Best CNN Training & Prediction Visualization Script)
This script uses the best hyperparameter configuration obtained from W&B Sweeps to train a CNN on the iNaturalist_12K dataset. It then visualizes the model's predictions on the test set using a clean image grid.

🧠 What It Does:
Defines a fixed configuration based on the best-performing hyperparameters from the sweep:

Activation: SiLU

Batch Normalization: Enabled

Dropout: 0.3

Learning Rate: 0.0004165

Conv Filters: [32, 64, 128, 128, 128]

Kernel Sizes: [3, 3, 3, 3, 3]

Dense Neurons: 256

Batch Size: 32

Data Augmentation: Enabled

Initializes the Nature12KDataModule with augmentation and prepares the data.

Instantiates the CNN model with the best configuration.

Trains the model using PyTorch Lightning, with W&B logging enabled under the project "best_model_eval" and run name "Best_CNN_Model".

Evaluates the trained model on the test dataset.

Visualizes predictions from the test set using visualize_predictions():

Displays 30 sample test images in a 10x3 grid.

Shows both true and predicted class names.

Saves the grid as test_predictions_grid.png.

🚀 How it works:

model, data_module = train_best_model()
visualize_predictions(model, data_module.test_dataloader(), data_module.class_names)
train_best_model() trains and tests the CNN with the best hyperparameters.

visualize_predictions() shows a grid of predicted vs. actual labels from the test set.
📓 train2.ipynb – Development Notebook for CNN + W&B on iNaturalist_12K
This notebook served as the central development workspace for implementing, testing, and refining the complete training pipeline for a CNN model on the iNaturalist_12K dataset using PyTorch Lightning and Weights & Biases (W&B).

🔧 Purpose:
train2.ipynb contains all code components in one place, including:

Dataset preprocessing and loading using Nature12KDataModule

CNN model architecture definition

W&B integration for experiment tracking

Hyperparameter sweep setup and launch

Best model training and evaluation

Prediction visualization

🧱 Modules Extracted From This Notebook:
The final codebase was later modularized and structured into clean Python scripts for production use:

Dataset.py → Contains the Nature12KDataModule

cnn_model.py → Contains the CNN class with flexible architecture

sweep_train.py → W&B Sweep script for automated tuning

best_model_train.py → Trains and evaluates CNN using best sweep config

train2.ipynb → The original development notebook

🧪 Features of the Notebook:
🎛️ End-to-end sweep setup with Bayesian optimization

📊 Logging to W&B dashboards

🔁 Supports multiple activation functions, batch norm, dropout, and data aug

📸 Visual analysis of model predictions on test images

✅ Manual override to retrain and evaluate with best hyperparameters

This notebook is great for:

Debugging and rapid prototyping

Visualizing training/test behavior interactively

Serving as a reference while developing modular code

PART-B:
ResNet.py
The code defines a PyTorch Lightning model (ResNetFinetune) for fine-tuning a ResNet architecture on a custom dataset. Here’s a summary of the key components:

Initialization (__init__):

The model uses a ResNet variant (e.g., resnet18) pretrained on ImageNet.

The final fully connected layer is replaced with one that outputs num_classes classes.

Options for freezing layers (freeze_type and freeze_upto_layer) are provided to allow partial training of the model.

Metrics for training, validation, and test phases are defined using torchmetrics (accuracy, precision, recall, F1 score).

Forward Pass (forward):

The input is passed through the ResNet backbone to get predictions.

Training Step (training_step):

The model computes the loss and various metrics (accuracy, precision, recall, F1 score) on the training set, logging the values.

Validation Step (validation_step):

Similar to the training step, but for the validation set. The loss and metrics are logged.

Test Step (test_step):

Similar to the validation step, but for the test set. The loss and metrics are logged.

Optimizer and Scheduler Configuration (configure_optimizers):

The model uses either Adam or SGD as the optimizer.

Optionally, a learning rate scheduler (StepLR) can be applied after training for a specified number of epochs.

Others files are same as PART-A Description.