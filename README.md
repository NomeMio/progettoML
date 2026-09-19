# Image Compression with Autoencoders (Track T8)

Machine Learning project, A.Y. 2024/2025 — Alexandru Nazare (student ID 0365256).

Train a convolutional autoencoder to compress flower images from the `102flowers` dataset, then compare the models (and their variants) against JPEG at several quality factors, in terms of **compression rate** and **reconstruction quality** (measured with SSIM).

Full write-up (in Italian): `documentazioneProgetto.pdf` / `documentazioneProgetto.ipynb`.

## Dataset and preprocessing

- ~8,200 flower images (102 species), 8.16 GB raw, with widely varying sizes (up to 1024x1168).
- Images are resized to a fixed **100x100 RGB** (0.24 GB); a 300x300 version was also generated (2.21 GB). 500x500 was dropped because of GPU memory limits.
- Pixels are scaled to [0, 1] and the alpha channel is removed.
- Data loading uses `keras.utils.image_dataset_from_directory` through the `getSamples` helper class in `scripts/commonUtils.py`.
- ~1% of the images plus a few random web images are kept aside in `testSet/`.

## Metrics and training

- **Quality:** SSIM, computed per channel and averaged over RGB.
- **Compression:** byte rate = `bytes / (pixels * 3)` (lower means stronger compression).
- **Loss:** custom `MSE * (1 + (1 - SSIM))`, which prioritizes MSE.
- **Optimizer:** Adam, lr = 0.001 (chosen with Keras-Tuner). Early stopping is the only regularization.
- Clustering (k-means/PCA, VGG16 features) to balance the dataset gave no useful results.

## Approach

The encoder produces a small "residual" image, which is stored as a **JPEG (quality=100)**. The decoder reconstructs the original image from it.

Since casting float to int8 has zero gradient and breaks training, training is done in stages:

1. Train the normal autoencoder, with float residuals.
2. Add a float-to-int "converter" layer after the encoder.
3. Freeze the encoder and retrain the decoder so it learns to work with the quantized residual.

Dense-layer and PCA-style autoencoders were tried but did not work well: their float residuals are too large, or they reach very low SSIM.

## Models

| Model | Description |
|---|---|
| **BW-RGB-50** | Two residuals: structure (50x50x1) and color (50x50x3) |
| **BW-RGB-25** | Structure (50x50x1) and color (25x25x3): lower byte rate, slightly lower SSIM |
| **RGB-50** | Single 50x50x3 residual using only downscaling/upscaling convolutions: simpler and weaker |

Weights are in `modelli/`, per-image results are in the `*_results.csv` files, and the architecture code is in the `modelli` module.

## Results

Mean SSIM and byte rate over the test set (approximate values from the plots):

| Method | Byte rate | SSIM |
|---|---|---|
| JPEG q=10 | ~0.05 | ~0.74 |
| JPEG q=50 | ~0.10 | ~0.92 |
| JPEG q=70 | ~0.12 | ~0.98 |
| JPEG q=90 | ~0.18 | ~0.99 |
| BW-RGB-25 | ~0.14 | ~0.81 |
| RGB-50 | ~0.14 | ~0.90 |
| BW-RGB-50 | ~0.19 | ~0.94 |

**Conclusion:** no autoencoder beats JPEG, which is expected given how simple the models are. BW-RGB-50 and RGB-50 give acceptable results but have a byte rate that is too high and an SSIM that is too low, respectively.

- Without the JPEG step (integer tensors only), BW-RGB-50 gives a fixed 2x size reduction (byte rate ~0.5) with SSIM ~0.93.
- Using JPEG 90 for the residual cuts the byte rate a lot but hurts SSIM. Retraining the decoder on it did not fix this.

## Project structure

```
documentazioneProgetto.ipynb / .pdf   project documentation and experiments
scripts/commonUtils.py                dataset, resizing, SSIM and plotting helpers
modelli/                              model definitions and trained weights (.h5)
testSet/                              test images
utilsImage/                           auxiliary images
*_results.csv                         per-image SSIM / byte rate results
```

## Requirements

Python, TensorFlow/Keras, scikit-image, pandas, NumPy, Pillow, and matplotlib (seaborn-style plots). A GPU is recommended.
