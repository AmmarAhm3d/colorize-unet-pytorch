# colorize-unet-pytorch

PyTorch U‑Net image colorization (LAB color space) — trained on COCO, includes evaluation (PSNR/SSIM/RMSE/SNR) and a Gradio demo.

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.0-black)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](/LICENSE)
[![Demo](https://img.shields.io/badge/Gradio-Demo-green)](#)

## 📋 Project Information

**Course:** Digital Image Processing  
**Project Type:** Deep Learning-based Image Colorization  
**Dataset:** MS COCO 2017 (Validation Set)  
**Training Duration:** ~2 hours on GPU  
**Final Training Loss:** 0.0717 (L1 Loss at Epoch 24)

---

## 🎯 Abstract

Automatic colorization of grayscale images is an ill-posed, one-to-many problem that has seen significant progress with deep learning. However, existing methods, particularly standard convolutional encoder-decoders like U-Net, often produce desaturated or spatially inconsistent colors due to an over-reliance on local pixel-level features. 

This project implements a **U-Net architecture from scratch** for automatic image colorization. The model converts grayscale (L*) images to color by predicting the color channels (a*, b*) in the LAB color space. We train the model on the COCO dataset using L1 loss for optimization and evaluate performance using comprehensive objective fidelity metrics including PSNR, SSIM, RMSE, and SNR.

---

## 🚀 The Problem We're Solving

The central challenge in automatic image colorization is not just predicting *a* color for each pixel, but predicting **plausible and globally consistent** colors that make sense for the entire scene.

### Common Failure Modes:
1. **Color Drabness:** Models predict "safe," desaturated colors because they represent statistical averages and minimize pixel-wise error.
2. **Color Inconsistency:** Large uniform regions (sky, lake) may be colored with different shades because decisions are made locally without global context.

### Our Approach:
- **Input:** Grayscale image (L* channel from LAB color space)
- **Output:** Two color channels (a*, b*) to reconstruct full-color image
- **Architecture:** U-Net with encoder-decoder structure and skip connections
- **Loss Function:** L1 Loss (Mean Absolute Error)
- **Color Space:** LAB (more perceptually uniform than RGB)

---

## 🏗️ Architecture Details

### U-Net Model Structure

```
Input: 1 channel (L - Grayscale) → Output: 2 channels (ab - Color)
```

**Encoder (Downsampling):**
- 6 convolutional blocks with BatchNorm and ReLU
- Progressive feature extraction: 64 → 128 → 256 → 512 → 512 → 512
- Kernel size: 4×4, Stride: 2, Padding: 1

**Bottleneck:**
- Final compression layer (512 channels)
- Captures highest-level abstract features

**Decoder (Upsampling):**
- 6 transposed convolutional blocks
- Skip connections from encoder for detail preservation
- Dropout (p=0.5) on first 3 decoder layers for regularization
- Progressive upsampling: 512 → 512 → 512 → 256 → 128 → 64

**Output Layer:**
- Final transposed convolution to 2 channels (a, b)
- Tanh activation to bound output to [-1, 1]

### Key Implementation Details

```python
class UNet(nn.Module):
    - Encoder: 6 downsampling blocks
    - Bottleneck: 512 channels at 4×4 resolution (for 256×256 input)
    - Decoder: 6 upsampling blocks with skip connections
    - Skip connections: Concatenation (not addition)
    - Dropout: Applied to first 3 decoder layers
    - Activation: ReLU (encoder/decoder), Tanh (output)
```

---

## 📊 Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-4 |
| Batch Size | 16 |
| Optimizer | Adam |
| Loss Function | L1 Loss (MAE) |
| Number of Epochs | 50 |
| Image Size | 256×256 |
| Dataset | COCO val2017 (~5k images) |
| Device | CUDA (GPU) |

### Training Process

1. **Data Preprocessing:**
   - RGB images → LAB color space
   - L channel normalized to [-1, 1] (from [0, 100])
   - ab channels normalized to [-1, 1] (from [-128, 127])

2. **Training Loop:**
   - Forward pass: Predict ab channels from L channel
   - Loss: L1 distance between predicted and true ab channels
   - Backpropagation with Adam optimizer
   - Checkpoint saved every epoch

3. **Results:**
   - Training completed in ~2 hours on GPU
   - Final loss: 0.0717 (Epoch 24 shown)
   - 50 model checkpoints saved
   - Visual samples saved every epoch

---

## 📈 Evaluation Metrics

We evaluate our model using **Objective Fidelity Criteria** as required:

### 1. PSNR (Peak Signal-to-Noise Ratio)
$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$

- **Range:** Typically 20-50 dB for images
- **Interpretation:**
  - > 40 dB: Excellent quality
  - 30-40 dB: Good quality
  - 20-30 dB: Acceptable quality
  - < 20 dB: Poor quality

### 2. SSIM (Structural Similarity Index)
- **Range:** [-1, 1], where 1 = identical images
- **Interpretation:**
  - > 0.9: Excellent similarity
  - 0.7-0.9: Good similarity
  - 0.5-0.7: Moderate similarity
  - < 0.5: Poor similarity

### 3. RMSE (Root Mean Square Error)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2}$$

- **Range:** [0, ∞]
- **Interpretation:** Lower is better (0 = perfect match)

### 4. SNR (Signal-to-Noise Ratio)
$$\text{SNR} = 10 \cdot \log_{10}\left(\frac{\text{Signal Power}}{\text{Noise Power}}\right)$$

- **Range:** Typically -10 to 50 dB
- **Interpretation:** Higher is better (> 20 dB is good)

---

## 📁 Project Structure

```
working/
├── model.py              # U-Net architecture implementation
├── data.py               # Dataset and data loading utilities
├── train.py              # Training script
├── app.py                # Gradio web interface for demo
├── evaluation.py         # Objective metrics evaluation
├── final_colorization_model.pth   # Final trained model
├── checkpoints/          # Model checkpoints (epochs 0-49)
│   ├── model_epoch_0.pth
│   ├── ...
│   └── model_epoch_49.pth
├── saved_images/         # Training progress visualizations
├── evaluation_results/   # Evaluation metrics and comparisons
└── coco/                 # COCO dataset
    └── images/val2017/
```

---

## 🛠️ Installation & Usage

### Prerequisites

```bash
pip install torch torchvision numpy pillow scikit-image matplotlib gradio tqdm
```

### Training the Model

```bash
python train.py
```

**Note:** Training has already been completed. The model is saved as `final_colorization_model.pth`.

### Evaluating the Model

Run objective fidelity metrics on test images:

```bash
python evaluation.py
```

This will:
- Evaluate 100 samples from the dataset
- Calculate PSNR, SSIM, RMSE, and SNR for each image
- Save comparison images showing: grayscale input, ground truth, and prediction
- Generate distribution plots for all metrics
- Save summary statistics to `evaluation_results/metrics_summary.txt`

### Running the Demo App

Launch the Gradio web interface:

```bash
python app.py
```

The app will:
- Start a web server with a user-friendly interface
- Allow you to upload grayscale images
- Display colorized results in real-time
- Provide a shareable link for demonstration

---

## 🎨 How It Works

### Color Space: LAB

We use **LAB color space** instead of RGB because:
- **L (Lightness):** 0-100, represents grayscale information
- **a (Green-Red):** -128 to 127, color opponent dimension
- **b (Blue-Yellow):** -128 to 127, color opponent dimension
- **Perceptually uniform:** Equal changes in LAB values correspond to equal perceptual differences
- **Separates luminance from color:** Perfect for colorization task

### Inference Pipeline

1. **Input:** Grayscale image (can be RGB converted to grayscale)
2. **Convert to LAB:** Extract L channel
3. **Normalize:** L channel to [-1, 1]
4. **Predict:** Model outputs ab channels [-1, 1]
5. **Denormalize:** Convert back to LAB range
6. **Reconstruct:** Combine L (original) + ab (predicted)
7. **Convert to RGB:** Final colorized image

---

## 🔬 Novel Contributions

While this implementation uses a standard U-Net architecture, our project demonstrates:

1. **From-Scratch Implementation:** Complete U-Net built without using pre-built libraries
2. **LAB Color Space Optimization:** Effective use of LAB for colorization
3. **Comprehensive Evaluation:** Rigorous objective fidelity metrics (PSNR, SSIM, RMSE, SNR)
4. **Production-Ready Deployment:** Gradio interface for real-world usage
5. **Complete Pipeline:** End-to-end solution from training to deployment

### Future Enhancements (Discussed but Not Implemented)

The **Learnable Color Palette** concept for globally-aware colorization:
- Attention mechanism at bottleneck to select scene-appropriate color palettes
- Global context injection into decoder
- Would address color drabness and inconsistency issues
- Lightweight alternative to GANs for vibrant colorization

---

## 📊 Results

### Training Performance
- **Final L1 Loss:** 0.0717 (Epoch 24)
- **Training Time:** ~2 hours on GPU
- **Convergence:** Stable training with consistent loss reduction

### Qualitative Results
The model successfully colorizes various scenes including:
- Natural landscapes (elephants, wildlife)
- Food items (strawberries, fruits)
- Complex textures and patterns

See `saved_images/` for training progress and `evaluation_results/` for detailed comparisons.

### Quantitative Results
Run `python evaluation.py` to generate comprehensive metrics including:
- Mean and standard deviation for all metrics
- Distribution plots
- Per-image comparisons with metric overlays

---

## 🎓 Academic Requirements Met

✅ **Deep Learning-Based:** U-Net CNN architecture  
✅ **No Traditional Methods:** No OpenCV colormaps or LUTs  
✅ **Custom Implementation:** Built from scratch  
✅ **Proper Dataset:** COCO 2017 (5k+ images)  
✅ **Objective Evaluation:** PSNR, SSIM, RMSE, SNR metrics  
✅ **Documentation:** Complete technical documentation  
✅ **Deployment:** Working Gradio demo application  

---

## 📚 References

### Core Concepts:
- **U-Net:** Originally proposed for biomedical image segmentation (Ronneberger et al., 2015)
- **Image Colorization:** Learning-based approach (Zhang et al., 2016)
- **LAB Color Space:** CIE 1976 L*a*b* color space for perceptual uniformity
- **Objective Fidelity Metrics:** Standard image quality assessment (Wang & Bovik, 2009)

### Technical Resources:
- PyTorch Documentation
- scikit-image for color space conversions
- COCO Dataset (Microsoft Common Objects in Context)

---

## 👥 Team Information

**Project:** Deep Learning Image Colorization  
**Members:**
- Ammar Ahmed
- Ahmad Naeem
- Khurram Imran
**Implementation:** Custom U-Net Architecture  
**Framework:** PyTorch  
**Date:** November 2025  

---

## 📝 License

This project is for academic purposes as part of a Digital Image Processing course.

---

## 🙏 Acknowledgments

- **COCO Dataset:** Microsoft Common Objects in Context
- **PyTorch:** Deep learning framework
- **Gradio:** Web interface framework
- **scikit-image:** Image processing utilities

---

**For questions or issues, please refer to the code documentation or contact the development team.**
