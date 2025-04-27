import os
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_nestedcv_model_accuracy(rf_results, xgb_results, save_path):
    # Extract accuracy scores and standard deviations
    models = ['Random Forest', 'XGBoost']
    accuracies = [rf_results['mean_accuracy'], xgb_results['mean_accuracy']]
    std_devs = [rf_results['std_accuracy'], xgb_results['std_accuracy']]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set bar positions
    bar_positions = np.arange(len(models))
    
    # Create bars
    bars = ax.bar(bar_positions, accuracies, yerr=std_devs, 
                  align='center', alpha=0.7, capsize=10,
                  color=['orange', 'green'], ecolor='black')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold', color = 'black',
                bbox=dict(facecolor='white', alpha=0.8, pad=3, edgecolor='none'))
    
    # Customize plot
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line for reference (e.g., at 0.5 for baseline)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Baseline (0.5)')
    
    # Add legend
    ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(save_path, format='png')  # Save the plot as a PNG file
    plt.close(fig)  # Close the plot to prevent it from displaying
    
    print(f"Plot saved to {save_path}")



def save_best_nestedcv_params(rf_best_params, xgb_best_params, save_path):
    # Create a dictionary to store both models' best parameters
    best_params = {
        'RandomForest': rf_best_params,
        'XGBoost': xgb_best_params
    }
    
    # Save the dictionary as a JSON file
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Best parameters saved to {save_path}")

# def save_best_nestedcv_params(rf_best_params, xgb_best_params, save_path):
#     with open(save_path, 'w') as f:
#         f.write("Best Random Forest parameters:\n")
#         for param, value in rf_best_params.items():
#             f.write(f"{param}: {value}\n")
        
#         f.write("\nBest XGBoost parameters:\n")
#         for param, value in xgb_best_params.items():
#             f.write(f"{param}: {value}\n")
        
#     print(f"Best parameters saved to {save_path}")