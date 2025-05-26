'''
Plotting script for RNN Model training metrics (Classification and Generator)
'''

from local_code.stage_4_code.Result_Loader import Result_Loader
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Path to result folder
RESULT_DIR = './result/stage_4_result/'

# Define metrics similar to stage 2
metrics = [
    ('accuracy_score', 'Accuracy'),
    ('f1_score', 'F1 Score'),
    ('precision_score', 'Precision'),
    ('recall_score', 'Recall')
]

# Colors for the plots
colors = ['blue', 'green', 'red', 'orange']

def plot_metrics():
    """Plot RNN training metrics similar to stage 2"""
    # Create subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    
    # Find all RNN results in the directory
    rnn_results = []
    for metric_name, _ in metrics:
        # Find files matching each metric
        pattern = f'RNN_Classification_prediction_result_1_{metric_name}'
        for file in os.listdir(RESULT_DIR):
            if pattern in file:
                rnn_results.append((metric_name, file))
    
    if not rnn_results:
        print(f"No RNN metric results found in {RESULT_DIR}")
        
        # If no specific metric files found, try to find any RNN file
        # This is a fallback in case all metrics are in one file
        any_rnn_files = [f for f in os.listdir(RESULT_DIR) if 'RNN' in f]
        if any_rnn_files:
            # Create a single plot for the first file found
            print(f"Found general RNN file: {any_rnn_files[0]}. Attempting to plot loss curve.")
            plot_single_loss_curve(any_rnn_files[0])
        return
    
    # Process each metric
    for idx, (metric_name, display_name) in enumerate(metrics):
        # Get the subplot position
        ax = axes[idx // 2, idx % 2]
        
        # Find the file for this metric
        matching_files = [f for m, f in rnn_results if m == metric_name]
        
        if not matching_files:
            ax.text(0.5, 0.5, f'No data found for {display_name}', 
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(f'{display_name}')
            continue
        
        # Load the data
        latest_file = matching_files[0]  # Take the first matching file
        loader = Result_Loader('result loader', '')
        loader.result_destination_folder_path = RESULT_DIR
        loader.result_destination_file_name = latest_file.split('.')[0]  # Remove extension
        
        try:
            loader.load()
            data = loader.data
            
            if not data or 'plotting_data' not in data:
                ax.text(0.5, 0.5, 'No plotting data found', 
                         horizontalalignment='center', verticalalignment='center')
                ax.set_title(f'{display_name}')
                continue
            
            plotting_data = data['plotting_data']
            
            # Check for required data
            if 'epoch' not in plotting_data or 'train_loss' not in plotting_data:
                ax.text(0.5, 0.5, 'Missing epoch or loss data', 
                         horizontalalignment='center', verticalalignment='center')
                ax.set_title(f'{display_name}')
                continue
            
            # Plot the loss curve
            ax.plot(plotting_data['epoch'], plotting_data['train_loss'], 
                    label=f'{display_name}', color=colors[idx])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{display_name}')
            ax.grid(True)
            ax.legend()
            
            # Add test score as annotation if available
            if 'true_y' in data and 'pred_y' in data:
                true_y = data['true_y']
                pred_y = data['pred_y']
                
                # Calculate the appropriate metric
                if metric_name == 'accuracy_score':
                    score = accuracy_score(true_y, pred_y)
                elif metric_name == 'f1_score':
                    score = f1_score(true_y, pred_y, average='binary')
                elif metric_name == 'precision_score':
                    score = precision_score(true_y, pred_y, average='binary')
                elif metric_name == 'recall_score':
                    score = recall_score(true_y, pred_y, average='binary')
                else:
                    score = 0  # Default
            
            
        except Exception as e:
            print(f"Error loading or plotting {display_name} data: {e}")
            ax.text(0.5, 0.5, f'Error loading data: {str(e)}', 
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(f'{display_name}')
    
    # Add overall title
    fig.suptitle('RNN Training Loss vs. Epoch by Metric', fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for title space
    
    # Save and display
    plt.savefig('rnn_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as 'rnn_metrics_comparison.png'")
    plt.show()

def plot_single_loss_curve(filename):
    """Plot a single loss curve when specific metric files aren't found"""
    loader = Result_Loader('RNN result loader', '')
    loader.result_destination_folder_path = RESULT_DIR
    loader.result_destination_file_name = filename.split('.')[0]  # Remove extension
    
    try:
        # Load the data
        loader.load()
        data = loader.data
        
        if not data or 'plotting_data' not in data:
            print("No plotting data found in the result!")
            return
        
        plotting_data = data['plotting_data']
        
        if 'epoch' not in plotting_data or 'train_loss' not in plotting_data:
            print("Required training data (epoch, loss) not found!")
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(plotting_data['epoch'], plotting_data['train_loss'], 'b-', marker='o', 
                 markersize=4, linewidth=2, label='Training Loss')
        
        plt.title('RNN Training Loss over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Add test accuracy if available
        if 'true_y' in data and 'pred_y' in data:
            test_acc = accuracy_score(data['true_y'], data['pred_y'])
        
        plt.tight_layout()
        plt.savefig('rnn_loss_curve.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as 'rnn_loss_curve.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error loading or plotting data: {e}")

def plot_generator_loss():
    """Plot RNN Generator training loss over epochs"""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Find generator result file
    generator_file = None
    for file in os.listdir(RESULT_DIR):
        if 'RNN_Generator_generation_result' in file:
            generator_file = os.path.join(RESULT_DIR, file)
            break
    
    if not generator_file:
        print(f"No RNN Generator results found in {RESULT_DIR}")
        return
    
    print(f"Loading generator results from: {generator_file}")
    
    # Load the results
    try:
        with open(generator_file, 'rb') as f:
            results = pickle.load(f)
        
        # Extract plotting data
        plotting_data = results.get('plotting_data', {})
        epochs = plotting_data.get('epoch', [])
        train_loss = plotting_data.get('train_loss', [])
        
        if not epochs or not train_loss:
            print("No training loss data found in results")
            return
        
        # Plot the loss curve
        ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
        
        # Add title and labels
        ax.set_title('RNN Generator Training Loss', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(RESULT_DIR, 'generator_loss_plot.png')
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"Error loading or plotting generator results: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot RNN model training metrics')
    parser.add_argument('--model', type=str, choices=['classification', 'generator', 'both'], 
                        default='both', help='Which model metrics to plot')
    args = parser.parse_args()
    
    # Plot based on user selection
    if args.model in ['classification', 'both']:
        print("Plotting RNN Classification metrics...")
        plot_metrics()
    
    if args.model in ['generator', 'both']:
        print("\nPlotting RNN Generator training loss...")
        plot_generator_loss()
    
    print("Done!")

if __name__ == "__main__":
    main()
