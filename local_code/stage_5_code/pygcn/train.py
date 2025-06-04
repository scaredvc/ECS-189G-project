from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pygcn.models import GCN
from local_code.stage_5_code.dataset_loader import Dataset_Loader

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = Dataset_Loader(dName="pubmed", dDescription="stage 5")
dataset.dataset_source_folder_path = "./data/stage_5_data/pubmed"
features, labels, adj, idx_train, idx_val, idx_test = dataset.convert_to_pygcn_format(dataset.load())

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# Lists to store training metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f1_scores = []
val_f1_scores = []
train_precisions = []
val_precisions = []
train_recalls = []
val_recalls = []


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    
    # Calculate metrics
    preds_train = output[idx_train].max(1)[1].cpu().numpy()
    labels_train = labels[idx_train].cpu().numpy()
    acc_train = accuracy_score(labels_train, preds_train)
    f1_train = f1_score(labels_train, preds_train, average='weighted')
    precision_train = precision_score(labels_train, preds_train, average='weighted')
    recall_train = recall_score(labels_train, preds_train, average='weighted')
    
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    # Calculate validation metrics
    preds_val = output[idx_val].max(1)[1].cpu().numpy()
    labels_val = labels[idx_val].cpu().numpy()
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy_score(labels_val, preds_val)
    f1_val = f1_score(labels_val, preds_val, average='weighted')
    precision_val = precision_score(labels_val, preds_val, average='weighted')
    recall_val = recall_score(labels_val, preds_val, average='weighted')
    
    # Store metrics for plotting
    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())
    train_accuracies.append(acc_train)
    val_accuracies.append(acc_val)
    train_f1_scores.append(f1_train)
    val_f1_scores.append(f1_val)
    train_precisions.append(precision_train)
    val_precisions.append(precision_val)
    train_recalls.append(recall_train)
    val_recalls.append(recall_val)
    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'f1_train: {:.4f}'.format(f1_train),
          'prec_train: {:.4f}'.format(precision_train),
          'rec_train: {:.4f}'.format(recall_train),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val),
          'f1_val: {:.4f}'.format(f1_val),
          'prec_val: {:.4f}'.format(precision_val),
          'rec_val: {:.4f}'.format(recall_val),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    
    # Calculate test metrics
    preds_test = output[idx_test].max(1)[1].cpu().numpy()
    labels_test = labels[idx_test].cpu().numpy()
    acc_test = accuracy_score(labels_test, preds_test)
    f1_test = f1_score(labels_test, preds_test, average='weighted')
    precision_test = precision_score(labels_test, preds_test, average='weighted')
    recall_test = recall_score(labels_test, preds_test, average='weighted')
    
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test),
          "f1_score= {:.4f}".format(f1_test),
          "precision= {:.4f}".format(precision_test),
          "recall= {:.4f}".format(recall_test))


def plot_training_metrics():
    """Plot training metrics similar to the provided image"""
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'GCN Training Loss vs. Epoch by Metric - {dataset.dataset_name.upper()} Dataset', fontsize=16)
    
    # Plot Training Loss for Accuracy metric
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Accuracy', linewidth=2)
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Training Loss for F1 Score metric
    axes[0, 1].plot(epochs, train_losses, 'g-', label='F1 Score', linewidth=2)
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot Training Loss for Precision metric
    axes[1, 0].plot(epochs, train_losses, 'r-', label='Precision', linewidth=2)
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot Training Loss for Recall metric
    axes[1, 1].plot(epochs, train_losses, 'orange', label='Recall', linewidth=2)
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'gcn_training_metrics_{dataset.dataset_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved as '{filename}'")
    plt.show()


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

# Plot and save training metrics
plot_training_metrics()

# Save training data to file
training_data = {
    'dataset': dataset.dataset_name,
    'epochs': list(range(1, len(train_losses) + 1)),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'train_f1_scores': train_f1_scores,
    'val_f1_scores': val_f1_scores,
    'train_precisions': train_precisions,
    'val_precisions': val_precisions,
    'train_recalls': train_recalls,
    'val_recalls': val_recalls
}

import json
filename = f'gcn_training_data_{dataset.dataset_name}.json'
with open(filename, 'w') as f:
    json.dump(training_data, f, indent=2)
print(f"Training data saved as '{filename}'")
