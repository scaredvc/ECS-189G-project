# ECS-189G Image Classification Project

## 📌 Introduction
This project implements various machine learning algorithms for image classification, with a focus on Convolutional Neural Networks (CNNs). It is designed with a modular structure that progresses through multiple stages, culminating in a CNN implementation that can classify different image datasets.

## 🔍 Project Structure
The project follows a well-organized, modular structure:

```
ECS-189G-project/  
├── local_code/  
│   ├── base_class/            # Abstract base classes  
│   ├── stage_1_code/          # Basic ML algorithms  
│   ├── stage_2_code/          # Intermediate implementations  
│   └── stage_3_code/          # Advanced CNN implementation  
└── script/  
    ├── stage_1_script/        # Entry points for stage 1  
    ├── stage_2_script/        # Entry points for stage 2  
    └── stage_3_script/        # Entry points for stage 3  
```

## 📋 Requirements
- Python 3.x
- PyTorch
- NumPy
- scikit-learn (for evaluation metrics)

## 🚀 Usage
To run the CNN implementation:

```bash
# Navigate to the script directory  
cd script/stage_3_script  
  
# Run the main script  
python main.py
```

The interactive TUI will guide you through the selection of:
- Dataset
- Evaluation metric
- Number of epochs
- Learning rate

## 📊 Available Datasets
The project supports multiple image datasets:

- **ORL** - Human face dataset with 40 people, 10 images each (9 for training, 1 for testing), 112x92x3 grayscale
- **MNIST** - Handwritten digit dataset with 60,000 training images and 10,000 testing images, 28x28 grayscale
- **CIFAR** - Colored object dataset with 50,000 training images and 10,000 testing images, 32x32x3 color

## 🧠 Model Architecture
### CNN Implementation
The project implements a Convolutional Neural Network with the following architecture:

- **Input Layer**: Accepts images with varying dimensions depending on the dataset
- **Convolutional Layer 1**: 32 filters with 3x3 kernel, ReLU activation, followed by 2x2 max pooling and dropout
- **Convolutional Layer 2**: 64 filters with 3x3 kernel, ReLU activation, adaptive pooling based on input dimensions, and dropout
- **Fully Connected Layer 1**: Maps to 128 nodes with ReLU activation and dropout
- **Output Layer**: Maps to the number of classes in the dataset

## ⚙️ Features
### Training Configuration
The system provides a text-based user interface (TUI) for configuring the training process:

- **Dataset Selection**: Choose between ORL, MNIST, or CIFAR datasets
- **Evaluation Metric**: Options include accuracy, F1 score, precision, or recall
- **Training Duration**: Select from different numbers of epochs (100, 200, 500, 1000)
- **Learning Rate**: Configure the model's learning rate (0.1, 0.01, 0.001, 0.0001)

### Training and Evaluation
- **Optimizer**: Uses Adam optimizer with weight decay for better generalization
- **Loss Function**: Standard cross-entropy loss for classification
- **Mini-batch Training**: Supports both full-batch and mini-batch training based on dataset size
- **GPU Acceleration**: Automatically utilizes CUDA if available

### Performance Monitoring
- Real-time training progress updates
- Performance metrics tracking
- Time-per-epoch measurement

## 🔬 Implementation Details
The project employs a structured approach based on object-oriented programming principles:

- **Base Classes**: Abstract classes define the interfaces for dataset, method, evaluation, result, and setting components
- **Concrete Implementations**: Stage-specific implementations extend base classes
- **Settings**: Support for different evaluation settings (K-fold CV, train-test split, etc.)

## 📝 Credits
This project appears to be developed for the ECS-189G course. The base framework was originally created by Jiawei Zhang.

## 🔗 Notes
- The repository is structured as a learning progression, with each stage building upon the previous one
- Stage 1 implements basic ML methods like Decision Trees and SVMs
- Stage 2 extends these implementations
- Stage 3 introduces CNN implementations for more complex image classification tasks

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/scaredvc/ECS-189G-project)