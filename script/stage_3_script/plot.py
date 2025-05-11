from local_code.stage_3_code.Result_Loader import Result_Loader
import matplotlib.pyplot as plt
import numpy as np

# result_folder should point to your Stage 3 results
result_folder = './result/stage_3_result/' 

# Define datasets to plot. Assumes files are named like CNN_CIFAR_prediction_result_1_accuracy_score
datasets = [
    ('CIFAR', 'CIFAR-10'), 
    ('MNIST', 'MNIST'),
    ('ORL', 'ORL')
]

# NOTE: The filenames found (e.g., CNN_CIFAR_prediction_result_1_accuracy_score)
# suggest 'accuracy_score' is the metric identifier in the file name.
# This script plots the 'loss' field from these files.
# Please verify these files contain the training loss data you intend to plot,
# especially given your comment about not using accuracy for CNN models.
metric_in_filename = 'accuracy_score' 

colors = ['blue', 'green', 'red', 'purple', 'orange'] # Added more colors just in case

num_datasets = len(datasets)
if num_datasets == 0:
    print("No datasets specified. Exiting.")
    exit()

# Adjust subplot layout based on number of datasets
active_axes_list = []
unused_axes_to_hide = []

if num_datasets == 3: # Special case for 2x2 layout with ORL on the second row, left position
    fig, grid_axes = plt.subplots(2, 2, figsize=(14, 10)) # Grid for 2x2
    # datasets order: CIFAR, MNIST, ORL
    active_axes_list = [grid_axes[0,0], grid_axes[0,1], grid_axes[1,0]]
    unused_axes_to_hide = [grid_axes[1,1]]
elif num_datasets == 1:
    fig, ax_single = plt.subplots(1, 1, figsize=(7, 5))
    active_axes_list = [ax_single]
elif num_datasets == 2: # Explicitly 1x2
    fig, ax_pair = plt.subplots(1, 2, figsize=(14, 5))
    active_axes_list = list(ax_pair) # ax_pair is a numpy array of axes
else: # Covers num_datasets >= 4 (and theoretically 0, but that's exited above)
    # Determine layout (e.g., 2 rows)
    ncols = (num_datasets + 1) // 2 
    nrows = 2
    if num_datasets == 5: ncols = 3 # Make it 2x3 for 5 plots, filling 5 of 6 cells
    # For num_datasets == 4, ncols = 2, so 2x2 grid, all cells used.
    # For num_datasets == 6, ncols = 3, so 2x3 grid, all cells used.
    
    fig, grid_axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    flattened_axes = list(grid_axes.flatten())
    active_axes_list = flattened_axes[:num_datasets]
    unused_axes_to_hide = flattened_axes[num_datasets:]

for idx, (dataset_prefix, dataset_label) in enumerate(datasets):
    if idx >= len(active_axes_list): # Safety check, should not be needed if logic above is correct
        print(f"Warning: Not enough subplots for dataset {dataset_label}. Skipping.")
        continue

    ax = active_axes_list[idx]
    loader = Result_Loader(f'{dataset_label} result loader', '')
    loader.result_destination_folder_path = result_folder
    file_name = f'CNN_{dataset_prefix}_prediction_result_1_{metric_in_filename}'
    loader.result_destination_file_name = file_name
    
    try:
        loader.load()
        if loader.data and 'plotting_data' in loader.data and \
           'epoch' in loader.data['plotting_data'] and 'loss' in loader.data['plotting_data']:
            plotting_data = loader.data['plotting_data']
            ax.plot(plotting_data['epoch'], plotting_data['loss'], label=f'{dataset_label} Loss', color=colors[idx % len(colors)])
            ax.set_xlabel('Epoch')
            ax.set_title(f'{dataset_label}')
            ax.grid(True)
            ax.legend()
            ax.set_ylabel('Loss') 
        else:
            missing_info = []
            if not loader.data: missing_info.append("data object")
            elif 'plotting_data' not in loader.data: missing_info.append("'plotting_data' in data")
            else:
                if 'epoch' not in loader.data['plotting_data']: missing_info.append("'epoch' in plotting_data")
                if 'loss' not in loader.data['plotting_data']: missing_info.append("'loss' in plotting_data")
            
            print(f"Warning: For {file_name}, 'plotting_data' (with 'epoch' and 'loss') not found or incomplete. Missing: {', '.join(missing_info)}.")
            ax.set_title(f'{dataset_label}\n(Data not found/valid)')
            ax.text(0.5, 0.5, 'Data not available or invalid', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_ylabel('Loss') 

    except Exception as e:
        print(f"Error loading or plotting data for {file_name}: {e}")
        ax.set_title(f'{dataset_label}\n(Error loading data)')
        ax.text(0.5, 0.5, 'Error loading data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_ylabel('Loss') 

# Hide any unused subplots if layout is larger than num_datasets
for ax_to_hide in unused_axes_to_hide:
    fig.delaxes(ax_to_hide)

fig.suptitle('CNN Training Loss vs. Epoch by Dataset', fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent suptitle overlap
plt.show()
