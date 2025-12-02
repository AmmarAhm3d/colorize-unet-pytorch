# evaluation.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from data import ColorizationDataset, lab_to_rgb_image
from model import UNet
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_rmse(original, predicted):
    """
    Calculate Root Mean Square Error (RMSE)
    """
    mse = np.mean((original - predicted) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_psnr(original, predicted, data_range=255):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    PSNR = 10 * log10(MAX^2 / MSE)
    Higher is better (typically 20-50 dB for images)
    """
    return peak_signal_noise_ratio(original, predicted, data_range=data_range)

def calculate_ssim(original, predicted, data_range=255):
    """
    Calculate Structural Similarity Index (SSIM)
    Range: [-1, 1], where 1 means identical images
    Typically > 0.9 is considered good
    """
    return structural_similarity(original, predicted, 
                                data_range=data_range, 
                                multichannel=True,
                                channel_axis=2)

def calculate_snr(original, predicted):
    """
    Calculate Signal-to-Noise Ratio (SNR)
    SNR = 10 * log10(signal_power / noise_power)
    """
    signal_power = np.mean(original ** 2)
    noise = original - predicted
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def evaluate_model(model_path, data_dir, num_samples=100, save_examples=True):
    """
    Evaluate the colorization model using objective fidelity metrics
    """
    print(f"Loading model from {model_path}")
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print(f"Loading dataset from {data_dir}")
    dataset = ColorizationDataset(data_dir, size=256)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Metrics storage
    psnr_scores = []
    ssim_scores = []
    rmse_scores = []
    snr_scores = []
    
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Evaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, total=num_samples)):
            if idx >= num_samples:
                break
            
            l_channel = batch['L'].to(DEVICE)
            ab_channels_true = batch['ab'].to(DEVICE)
            
            # Predict ab channels
            ab_channels_pred = model(l_channel)
            
            # Convert to RGB for metrics calculation
            original_rgb = lab_to_rgb_image(l_channel[0], ab_channels_true[0])
            predicted_rgb = lab_to_rgb_image(l_channel[0], ab_channels_pred[0])
            
            # Calculate metrics
            psnr = calculate_psnr(original_rgb, predicted_rgb)
            ssim = calculate_ssim(original_rgb, predicted_rgb)
            rmse = calculate_rmse(original_rgb.astype(np.float32), 
                                 predicted_rgb.astype(np.float32))
            snr = calculate_snr(original_rgb.astype(np.float32), 
                               predicted_rgb.astype(np.float32))
            
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            rmse_scores.append(rmse)
            snr_scores.append(snr)
            
            # Save first 10 examples
            if save_examples and idx < 10:
                save_comparison(l_channel[0], ab_channels_true[0], ab_channels_pred[0], 
                              idx, results_dir, psnr, ssim, rmse, snr)
    
    # Calculate statistics
    results = {
        'PSNR': {
            'mean': np.mean(psnr_scores),
            'std': np.std(psnr_scores),
            'min': np.min(psnr_scores),
            'max': np.max(psnr_scores)
        },
        'SSIM': {
            'mean': np.mean(ssim_scores),
            'std': np.std(ssim_scores),
            'min': np.min(ssim_scores),
            'max': np.max(ssim_scores)
        },
        'RMSE': {
            'mean': np.mean(rmse_scores),
            'std': np.std(rmse_scores),
            'min': np.min(rmse_scores),
            'max': np.max(rmse_scores)
        },
        'SNR': {
            'mean': np.mean(snr_scores),
            'std': np.std(snr_scores),
            'min': np.min(snr_scores),
            'max': np.max(snr_scores)
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print("OBJECTIVE FIDELITY METRICS EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples evaluated: {num_samples}")
    print("-"*60)
    
    for metric, stats in results.items():
        print(f"\n{metric}:")
        print(f"  Mean:  {stats['mean']:.4f}")
        print(f"  Std:   {stats['std']:.4f}")
        print(f"  Min:   {stats['min']:.4f}")
        print(f"  Max:   {stats['max']:.4f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE:")
    print("-"*60)
    print("PSNR (Peak Signal-to-Noise Ratio):")
    print("  > 40 dB: Excellent quality")
    print("  30-40 dB: Good quality")
    print("  20-30 dB: Acceptable quality")
    print("  < 20 dB: Poor quality")
    print("\nSSIM (Structural Similarity Index):")
    print("  > 0.9: Excellent similarity")
    print("  0.7-0.9: Good similarity")
    print("  0.5-0.7: Moderate similarity")
    print("  < 0.5: Poor similarity")
    print("\nRMSE (Root Mean Square Error):")
    print("  Lower is better (0 = perfect match)")
    print("\nSNR (Signal-to-Noise Ratio):")
    print("  Higher is better (> 20 dB is good)")
    print("="*60 + "\n")
    
    # Save results to file
    with open(os.path.join(results_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("OBJECTIVE FIDELITY METRICS EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write("-"*60 + "\n\n")
        
        for metric, stats in results.items():
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std:  {stats['std']:.4f}\n")
            f.write(f"  Min:  {stats['min']:.4f}\n")
            f.write(f"  Max:  {stats['max']:.4f}\n\n")
    
    # Create visualization
    plot_metrics_distribution(psnr_scores, ssim_scores, rmse_scores, snr_scores, results_dir)
    
    return results

def save_comparison(l_channel, ab_true, ab_pred, idx, save_dir, psnr, ssim, rmse, snr):
    """
    Save side-by-side comparison of original and predicted images
    """
    original_rgb = lab_to_rgb_image(l_channel, ab_true)
    predicted_rgb = lab_to_rgb_image(l_channel, ab_pred)
    grayscale = lab_to_rgb_image(l_channel, torch.zeros_like(ab_true))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(grayscale)
    axes[0].set_title('Input (Grayscale)', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(original_rgb)
    axes[1].set_title('Ground Truth (Original)', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(predicted_rgb)
    axes[2].set_title('Predicted (Colorized)', fontsize=12)
    axes[2].axis('off')
    
    fig.suptitle(f'Sample {idx} - PSNR: {psnr:.2f}dB | SSIM: {ssim:.4f} | RMSE: {rmse:.2f} | SNR: {snr:.2f}dB', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_{idx:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_metrics_distribution(psnr_scores, ssim_scores, rmse_scores, snr_scores, save_dir):
    """
    Plot distribution of all metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PSNR
    axes[0, 0].hist(psnr_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(psnr_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(psnr_scores):.2f}')
    axes[0, 0].set_xlabel('PSNR (dB)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('PSNR Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # SSIM
    axes[0, 1].hist(ssim_scores, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(ssim_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(ssim_scores):.4f}')
    axes[0, 1].set_xlabel('SSIM', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('SSIM Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # RMSE
    axes[1, 0].hist(rmse_scores, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(np.mean(rmse_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(rmse_scores):.2f}')
    axes[1, 0].set_xlabel('RMSE', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('RMSE Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # SNR
    axes[1, 1].hist(snr_scores, bins=30, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(np.mean(snr_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(snr_scores):.2f}')
    axes[1, 1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('SNR Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nMetrics distribution plot saved to {save_dir}/metrics_distribution.png")

if __name__ == "__main__":
    # Evaluate the final trained model
    MODEL_PATH = "final_colorization_model.pth"
    DATA_DIR = "coco/images/val2017"
    NUM_SAMPLES = 100  # Adjust based on how many images you want to evaluate
    
    results = evaluate_model(MODEL_PATH, DATA_DIR, NUM_SAMPLES, save_examples=True)
