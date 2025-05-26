# ECS-189G Deep Learning Project

## 📌 Introduction
This project implements various machine learning algorithms for different tasks, progressing through multiple stages:
- **Stage 1-2**: Basic and intermediate ML algorithms
- **Stage 3**: Convolutional Neural Networks (CNNs) for image classification
- **Stage 4**: Recurrent Neural Networks (RNNs) for text classification and text generation

## 🔍 Project Structure
The project follows a well-organized, modular structure:

```
ECS-189G-project/  
├── local_code/  
│   ├── base_class/            # Abstract base classes  
│   ├── stage_1_code/          # Basic ML algorithms  
│   ├── stage_2_code/          # Intermediate implementations  
│   ├── stage_3_code/          # Advanced CNN implementation  
│   └── stage_4_code/          # RNN for text classification and generation  
└── script/  
    ├── stage_1_script/        # Entry points for stage 1  
    ├── stage_2_script/        # Entry points for stage 2  
    ├── stage_3_script/        # Entry points for stage 3  
    └── stage_4_script/        # Entry points for stage 4  
```

## 🚀 Usage

### CNN Implementation (Stage 3)
```bash
pip install -r requirements.txt
python -m script.stage_3_script.main
```

### RNN Implementation (Stage 4)
```bash
pip install -r requirements.txt
python -m script.stage_4_script.main
```

The interactive TUI will guide you through the selection of:
- Model type (classification or generation for Stage 4)
- Evaluation metric
- Number of epochs
- Learning rate
- Temperature (for text generation)

### Plotting Results (Stage 4)
```bash
# Plot both classification and generation results
python -m script.stage_4_script.plot

# Plot only classification results
python -m script.stage_4_script.plot --model classification

# Plot only generator results
python -m script.stage_4_script.plot --model generator
```

## 📊 Available Datasets

### Image Datasets (Stage 3)
- **ORL** - Human face dataset with 40 people, 10 images each (9 for training, 1 for testing), 112x92x3 grayscale
- **MNIST** - Handwritten digit dataset with 60,000 training images and 10,000 testing images, 28x28 grayscale
- **CIFAR** - Colored object dataset with 50,000 training images and 10,000 testing images, 32x32x3 color

### Text Datasets (Stage 4)
- **Text Classification** - Dataset of jokes with positive and negative sentiment labels
- **Joke Generation** - Dataset of jokes for training a text generation model

## 🧠 Model Architectures

### CNN Implementation (Stage 3)
The project implements a Convolutional Neural Network with the following architecture:

- **Input Layer**: Accepts images with varying dimensions depending on the dataset
- **Convolutional Layer 1**: 32 filters with 3x3 kernel, ReLU activation, followed by 2x2 max pooling and dropout
- **Convolutional Layer 2**: 64 filters with 3x3 kernel, ReLU activation, adaptive pooling based on input dimensions, and dropout
- **Fully Connected Layer 1**: Maps to 128 nodes with ReLU activation and dropout
- **Output Layer**: Maps to the number of classes in the dataset

### RNN Classification (Stage 4)
The text classification RNN model features:

- **Embedding Layer**: Maps words to 100-dimensional vectors
- **LSTM Layer**: Hidden dimension of 100 with bidirectional processing
- **Fully Connected Layer**: Maps to binary classification output
- **Training**: Uses Adam optimizer with gradient clipping to prevent exploding gradients

### RNN Generator (Stage 4)
The joke generation RNN model features:

- **Embedding Layer**: Maps words to 128-dimensional vectors
- **LSTM Layer**: Hidden dimension of 256 with 2 layers for better sequence modeling
- **Temperature Sampling**: Controls randomness in generation (lower values = more predictable output)
- **Input Processing**: Takes the first 5 words of a joke as input and generates the rest
- **Generation Strategy**: Uses top-k sampling for more diverse and interesting joke completions

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

## 🔗 Notes
- The repository is structured as a learning progression, with each stage building upon the previous one
- Stage 1 implements basic ML methods like Decision Trees and SVMs
- Stage 2 extends these implementations
- Stage 3 introduces CNN implementations for more complex image classification tasks

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/scaredvc/ECS-189G-project)