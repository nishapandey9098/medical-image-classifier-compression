# Compressing a Medical Image Classifier for Offline Use: FP16, INT8 and Pruning Trade-offs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nishapandey9098/medical-image-classifier-compression/blob/main/Project3_Model_Compression.ipynb)

> A small, reproducible benchmark study on public low-resolution datasets. **Not validated for clinical use.**

## 1. Research question

When a retinal / chest X-ray classifier has to run **offline on a cheap device**, how much **size** and **CPU latency** can be saved with FP16, INT8 quantization and pruning, and what does it cost in **accuracy** and **calibration**?

## 2. Models and data

- **Models:** the two ResNet-18 reference models from Project 1 ([link to your Project 1 repository]): PneumoniaMNIST = `pretrained+aug`, RetinaMNIST = `pretrained+aug+clahe`. One trained model per dataset (seed 0).
- **Data:** PneumoniaMNIST and RetinaMNIST from [MedMNIST](https://medmnist.com/), 64×64 pixels, official validation / test split. The *training* images were used only to calibrate INT8 and to fine-tune pruned models.

## 3. Variants

| Variant | How it was made |
|---|---|
| FP32 | The Project 1 model exported to ONNX (opset 17, batch size 1) |
| FP16 | ONNX weights converted to 16-bit floats (input and output kept FP32) |
| INT8 | Post-training **static** quantization (ONNX Runtime, QDQ format, per-channel weights, 128 calibration images from the training set) |
| unstr 50 / 70 / 90% | Global magnitude pruning of the convolution weights (first and last layer excluded), as one-shot and as +FT (fine-tuned 8 epochs, learning rate 2e-4, epoch chosen by validation loss) |
| struct 25 / 50 / 75% | Removal of the least important filters (L1 norm) inside every residual block (first convolution of the block and the matching input channels of the second), one-shot and +FT |
| struct 50% +FT +INT8 | The fine-tuned structured model, then INT8 quantization |

## 4. Methods

- **Speed test:** ONNX Runtime 1.32.0 on CPU (`Intel(R) Xeon(R) CPU @ 2.00GHz`), **1 thread**, **batch size 1**, the same test image for every model. Untimed warm-up (20 runs), then 5 rounds × 40 timed runs per model; models are timed in a **different random order in every round** (interleaved). Reported: **median** latency, IQR, p95; speed-up = median FP32 / median variant with a 95% bootstrap interval over the timings.
- **Size:** file size in MB and size after deflate compression (zlib level 6).
- **Quality:** accuracy, AUC (PneumoniaMNIST) or QWK (RetinaMNIST) (the Project 1 main scores), ECE (15 bins) and NLL on the test set; agreement with the FP32 predictions; counts of predictions that were fixed or broken.
- **Changes vs FP32:** paired **bootstrap over the test images** (1000 redraws; both models scored on the same redrawn images); 95% percentile interval. This captures the luck of *which test images we have*, not training-run variation.
- **Choosing a variant:** only deployable variants whose **validation** score is at most 0.01 (PneumoniaMNIST) or 0.03 (RetinaMNIST) below FP32 qualify; the fastest and the smallest of them are recommended; test results are read only afterwards.

## 5. Results

All numbers are on the **test set** (the validation score is used only by the recommendation rule). CPU: Intel(R) Xeon(R) CPU @ 2.00GHz (avx2, avx512f); 1 thread; batch size 1; ONNX Runtime 1.32.0.

### PneumoniaMNIST

#### Table 1: size, speed, accuracy, calibration

| variant | stage | size (MB) | zip (MB) | non-zero weights (M) | zero weights (%) | latency (ms) | IQR (ms) | p95 (ms) | speed-up [95% CI] | accuracy | AUC | ECE | NLL | agree with FP32 (%) | val AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP32 | original | 44.701 | 41.541 | 11.168 | 0.000 | 6.070 | 1.561 | 7.632 | 1.00× [0.95, 1.06] | 0.913 | 0.985 | 0.065 | 0.334 | 100.000 | 0.998 |
| FP16 | post-training | 22.358 | 20.714 | 11.168 | 0.000 | 7.699 | 1.286 | 9.971 | 0.79× [0.77, 0.84] | 0.913 | 0.985 | 0.065 | 0.334 | 100.000 | 0.998 |
| INT8 | post-training | 11.295 | 9.411 | 11.168 | 0.000 | 3.193 | 1.053 | 3.902 | 1.90× [1.83, 2.05] | 0.824 | 0.979 | 0.146 | 0.748 | 90.385 | 0.997 |
| unstr 50% one-shot | one-shot | 44.701 | – | 5.589 | 49.953 | – | – | – | – | 0.878 | 0.984 | 0.085 | 0.414 | 96.154 | 0.998 |
| unstr 50% +FT | fine-tuned | 44.701 | 24.891 | 5.589 | 49.953 | 6.458 | 2.034 | 8.307 | 0.94× [0.86, 1.04] | 0.873 | 0.982 | 0.102 | 0.507 | 95.032 | 0.998 |
| unstr 70% one-shot | one-shot | 44.701 | – | 3.358 | 69.935 | – | – | – | – | 0.716 | 0.984 | 0.233 | 0.921 | 79.647 | 0.998 |
| unstr 70% +FT | fine-tuned | 44.701 | 16.868 | 3.358 | 69.935 | 6.520 | 1.662 | 8.348 | 0.93× [0.91, 0.99] | 0.888 | 0.983 | 0.078 | 0.394 | 95.513 | 0.997 |
| unstr 90% one-shot | one-shot | 44.701 | – | 1.126 | 89.916 | – | – | – | – | 0.625 | 0.918 | 0.372 | 2.160 | 70.513 | 0.976 |
| unstr 90% +FT | fine-tuned | 44.701 | 7.937 | 1.126 | 89.916 | 6.860 | 2.186 | 9.004 | 0.88× [0.84, 0.94] | 0.905 | 0.979 | 0.071 | 0.297 | 97.276 | 0.997 |
| struct 25% one-shot | one-shot | 33.713 | – | 8.422 | 0.000 | – | – | – | – | 0.659 | 0.987 | 0.305 | 1.166 | 73.878 | 0.997 |
| struct 25% +FT | fine-tuned | 33.713 | 31.337 | 8.422 | 0.000 | 4.986 | 1.853 | 7.366 | 1.22× [1.15, 1.40] | 0.926 | 0.985 | 0.054 | 0.282 | 97.756 | 0.999 |
| struct 50% one-shot | one-shot | 22.726 | – | 5.675 | 0.000 | – | – | – | – | 0.625 | 0.618 | 0.370 | 2.005 | 70.513 | 0.616 |
| struct 50% +FT | fine-tuned | 22.726 | 21.120 | 5.675 | 0.000 | 2.723 | 1.088 | 4.306 | 2.23× [1.84, 2.40] | 0.905 | 0.985 | 0.063 | 0.290 | 97.596 | 0.998 |
| struct 75% one-shot | one-shot | 11.738 | – | 2.929 | 0.000 | – | – | – | – | 0.625 | 0.850 | 0.368 | 1.874 | 70.513 | 0.854 |
| struct 75% +FT | fine-tuned | 11.738 | 10.913 | 2.929 | 0.000 | 1.521 | 0.510 | 2.117 | 3.99× [3.68, 4.26] | 0.913 | 0.974 | 0.047 | 0.252 | 94.231 | 0.997 |
| struct 50% +FT +INT8 | fine-tuned + quantized | 5.786 | 4.889 | 5.675 | 0.000 | 1.564 | 0.533 | 2.260 | 3.88× [3.58, 4.13] | 0.843 | 0.980 | 0.121 | 0.519 | 91.987 | 0.997 |

*Speed-up = median FP32 latency / median variant latency, with a 95% bootstrap interval over the timings. – = not measured (only the fine-tuned version of each pruned model was timed and compressed). `agree with FP32` = share of test images with the same predicted class as FP32; `val AUC` = validation score used by the recommendation rule.*

#### Table 2: change vs FP32 with 95% bootstrap intervals over the test images

| variant | accuracy | AUC | ECE | NLL | clearly better on | clearly worse on | changed | fixed | broken |
|---|---|---|---|---|---|---|---|---|---|
| FP16 | +0.000 [+0.000, +0.000] | +0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.000] | - | - | 0 | 0 | 0 |
| INT8 | -0.090 [-0.115, -0.067] | -0.006 [-0.013, +0.001] | +0.080 [+0.060, +0.102] | +0.414 [+0.333, +0.500] | - | accuracy, ECE, NLL | 60 | 2 | 58 |
| unstr 50% one-shot | -0.035 [-0.051, -0.021] | -0.001 [-0.003, +0.000] | +0.020 [+0.007, +0.033] | +0.080 [+0.060, +0.101] | - | accuracy, ECE, NLL | 24 | 1 | 23 |
| unstr 50% +FT | -0.040 [-0.058, -0.022] | -0.003 [-0.006, +0.001] | +0.037 [+0.022, +0.055] | +0.174 [+0.127, +0.228] | - | accuracy, ECE, NLL | 31 | 3 | 28 |
| unstr 70% one-shot | -0.197 [-0.232, -0.165] | -0.000 [-0.002, +0.002] | +0.167 [+0.135, +0.194] | +0.588 [+0.501, +0.679] | - | accuracy, ECE, NLL | 127 | 2 | 125 |
| unstr 70% +FT | -0.026 [-0.043, -0.010] | -0.002 [-0.007, +0.003] | +0.012 [-0.004, +0.026] | +0.060 [+0.019, +0.105] | - | accuracy, NLL | 28 | 6 | 22 |
| unstr 90% one-shot | -0.288 [-0.325, -0.252] | -0.066 [-0.085, -0.049] | +0.307 [+0.267, +0.342] | +1.826 [+1.626, +2.045] | - | accuracy, AUC, ECE, NLL | 184 | 2 | 182 |
| unstr 90% +FT | -0.008 [-0.021, +0.005] | -0.005 [-0.014, +0.002] | +0.006 [-0.008, +0.015] | -0.036 [-0.075, +0.003] | - | - | 17 | 6 | 11 |
| struct 25% one-shot | -0.255 [-0.292, -0.220] | +0.003 [-0.002, +0.010] | +0.240 [+0.203, +0.272] | +0.833 [+0.715, +0.948] | - | accuracy, ECE, NLL | 163 | 2 | 161 |
| struct 25% +FT | +0.013 [+0.002, +0.024] | +0.000 [-0.003, +0.004] | -0.011 [-0.023, +0.001] | -0.051 [-0.080, -0.023] | accuracy, NLL | - | 14 | 11 | 3 |
| struct 50% one-shot | -0.288 [-0.325, -0.252] | -0.366 [-0.412, -0.320] | +0.305 [+0.265, +0.340] | +1.671 [+1.484, +1.872] | - | accuracy, AUC, ECE, NLL | 184 | 2 | 182 |
| struct 50% +FT | -0.008 [-0.021, +0.003] | -0.000 [-0.004, +0.005] | -0.002 [-0.014, +0.010] | -0.043 [-0.080, -0.007] | NLL | - | 15 | 5 | 10 |
| struct 75% one-shot | -0.288 [-0.325, -0.252] | -0.135 [-0.172, -0.104] | +0.303 [+0.263, +0.338] | +1.541 [+1.362, +1.730] | - | accuracy, AUC, ECE, NLL | 184 | 2 | 182 |
| struct 75% +FT | +0.000 [-0.019, +0.019] | -0.011 [-0.017, -0.005] | -0.018 [-0.035, -0.001] | -0.082 [-0.133, -0.032] | ECE, NLL | AUC | 36 | 18 | 18 |
| struct 50% +FT +INT8 | -0.071 [-0.091, -0.050] | -0.004 [-0.010, +0.001] | +0.055 [+0.034, +0.074] | +0.186 [+0.131, +0.238] | - | accuracy, ECE, NLL | 50 | 3 | 47 |

*Each cell is the change vs FP32 [95% bootstrap interval over the test images]. `changed` = test predictions that differ from FP32; `fixed` = wrong in FP32 and right now; `broken` = right in FP32 and wrong now.*

Recommended by validation score (tolerance 0.01): fastest = `struct 75% +FT`, smallest = `struct 50% +FT +INT8`.

### RetinaMNIST

#### Table 1: size, speed, accuracy, calibration

| variant | stage | size (MB) | zip (MB) | non-zero weights (M) | zero weights (%) | latency (ms) | IQR (ms) | p95 (ms) | speed-up [95% CI] | accuracy | QWK | ECE | NLL | agree with FP32 (%) | val QWK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP32 | original | 44.707 | 41.526 | 11.169 | 0.000 | 6.176 | 0.837 | 7.419 | 1.00× [0.98, 1.02] | 0.557 | 0.564 | 0.132 | 1.275 | 100.000 | 0.686 |
| FP16 | post-training | 22.362 | 20.700 | 11.169 | 0.000 | 6.533 | 2.109 | 9.182 | 0.95× [0.91, 0.98] | 0.555 | 0.563 | 0.134 | 1.275 | 99.750 | 0.686 |
| INT8 | post-training | 11.296 | 9.407 | 11.169 | 0.000 | 2.597 | 0.404 | 3.304 | 2.38× [2.31, 2.42] | 0.565 | 0.553 | 0.130 | 1.237 | 88.000 | 0.658 |
| unstr 50% one-shot | one-shot | 44.707 | – | 5.591 | 49.946 | – | – | – | – | 0.537 | 0.558 | 0.170 | 1.294 | 85.750 | 0.667 |
| unstr 50% +FT | fine-tuned | 44.707 | 24.859 | 5.591 | 49.946 | 6.140 | 1.279 | 8.144 | 1.01× [0.97, 1.03] | 0.565 | 0.599 | 0.207 | 1.407 | 70.000 | 0.671 |
| unstr 70% one-shot | one-shot | 44.707 | – | 3.359 | 69.925 | – | – | – | – | 0.552 | 0.506 | 0.162 | 1.372 | 70.250 | 0.602 |
| unstr 70% +FT | fine-tuned | 44.707 | 16.851 | 3.359 | 69.925 | 6.117 | 1.166 | 7.879 | 1.01× [0.98, 1.03] | 0.527 | 0.567 | 0.175 | 1.305 | 72.750 | 0.663 |
| unstr 90% one-shot | one-shot | 44.707 | – | 1.128 | 89.904 | – | – | – | – | 0.435 | 0.000 | 0.497 | 3.214 | 45.000 | 0.000 |
| unstr 90% +FT | fine-tuned | 44.707 | 7.937 | 1.128 | 89.904 | 5.702 | 1.232 | 8.064 | 1.08× [1.03, 1.11] | 0.573 | 0.643 | 0.081 | 1.155 | 65.500 | 0.690 |
| struct 25% one-shot | one-shot | 33.719 | – | 8.423 | 0.000 | – | – | – | – | 0.398 | 0.380 | 0.120 | 1.480 | 49.000 | 0.503 |
| struct 25% +FT | fine-tuned | 33.719 | 31.328 | 8.423 | 0.000 | 4.370 | 0.827 | 5.375 | 1.41× [1.37, 1.45] | 0.540 | 0.512 | 0.223 | 1.503 | 68.500 | 0.661 |
| struct 50% one-shot | one-shot | 22.732 | – | 5.677 | 0.000 | – | – | – | – | 0.435 | 0.000 | 0.270 | 1.636 | 45.000 | 0.000 |
| struct 50% +FT | fine-tuned | 22.732 | 21.129 | 5.677 | 0.000 | 2.702 | 0.351 | 3.552 | 2.29× [2.22, 2.32] | 0.547 | 0.550 | 0.156 | 1.290 | 64.250 | 0.702 |
| struct 75% one-shot | one-shot | 11.745 | – | 2.930 | 0.000 | – | – | – | – | 0.435 | 0.000 | 0.328 | 1.821 | 45.000 | 0.000 |
| struct 75% +FT | fine-tuned | 11.745 | 10.917 | 2.930 | 0.000 | 1.569 | 0.260 | 2.057 | 3.94× [3.80, 4.02] | 0.530 | 0.557 | 0.076 | 1.124 | 60.250 | 0.651 |
| struct 50% +FT +INT8 | fine-tuned + quantized | 5.787 | 4.888 | 5.677 | 0.000 | 1.559 | 0.582 | 2.364 | 3.96× [3.66, 4.17] | 0.537 | 0.547 | 0.152 | 1.237 | 64.750 | 0.727 |

*Speed-up = median FP32 latency / median variant latency, with a 95% bootstrap interval over the timings. – = not measured (only the fine-tuned version of each pruned model was timed and compressed). `agree with FP32` = share of test images with the same predicted class as FP32; `val QWK` = validation score used by the recommendation rule.*

#### Table 2: change vs FP32 with 95% bootstrap intervals over the test images

| variant | accuracy | QWK | ECE | NLL | clearly better on | clearly worse on | changed | fixed | broken |
|---|---|---|---|---|---|---|---|---|---|
| FP16 | -0.002 [-0.008, +0.000] | -0.001 [-0.004, +0.000] | +0.002 [-0.005, +0.005] | +0.000 [-0.000, +0.000] | - | - | 1 | 0 | 1 |
| INT8 | +0.007 [-0.018, +0.037] | -0.011 [-0.045, +0.026] | -0.003 [-0.029, +0.034] | -0.037 [-0.088, +0.008] | - | - | 48 | 17 | 14 |
| unstr 50% one-shot | -0.020 [-0.050, +0.007] | -0.007 [-0.064, +0.049] | +0.037 [-0.002, +0.061] | +0.019 [-0.032, +0.068] | - | - | 57 | 13 | 21 |
| unstr 50% +FT | +0.007 [-0.033, +0.047] | +0.034 [-0.032, +0.108] | +0.075 [+0.024, +0.114] | +0.132 [+0.030, +0.228] | - | ECE, NLL | 120 | 40 | 37 |
| unstr 70% one-shot | -0.005 [-0.050, +0.040] | -0.058 [-0.126, +0.003] | +0.030 [-0.019, +0.072] | +0.097 [+0.009, +0.186] | - | NLL | 119 | 41 | 43 |
| unstr 70% +FT | -0.030 [-0.070, +0.010] | +0.003 [-0.060, +0.064] | +0.042 [-0.005, +0.081] | +0.031 [-0.058, +0.115] | - | - | 109 | 28 | 40 |
| unstr 90% one-shot | -0.122 [-0.182, -0.062] | -0.564 [-0.637, -0.486] | +0.364 [+0.302, +0.416] | +1.940 [+1.625, +2.271] | - | accuracy, QWK, ECE, NLL | 220 | 48 | 97 |
| unstr 90% +FT | +0.015 [-0.032, +0.057] | +0.078 [+0.008, +0.148] | -0.052 [-0.086, +0.000] | -0.119 [-0.210, -0.028] | QWK, NLL | - | 138 | 49 | 43 |
| struct 25% one-shot | -0.160 [-0.215, -0.105] | -0.184 [-0.275, -0.100] | -0.012 [-0.057, +0.040] | +0.205 [+0.096, +0.316] | - | accuracy, QWK, NLL | 204 | 37 | 101 |
| struct 25% +FT | -0.017 [-0.060, +0.025] | -0.052 [-0.132, +0.023] | +0.091 [+0.040, +0.128] | +0.228 [+0.115, +0.346] | - | ECE, NLL | 126 | 34 | 41 |
| struct 50% one-shot | -0.122 [-0.182, -0.062] | -0.564 [-0.637, -0.486] | +0.138 [+0.076, +0.190] | +0.362 [+0.221, +0.507] | - | accuracy, QWK, ECE, NLL | 220 | 48 | 97 |
| struct 50% +FT | -0.010 [-0.057, +0.037] | -0.014 [-0.097, +0.070] | +0.023 [-0.022, +0.067] | +0.015 [-0.093, +0.122] | - | - | 143 | 43 | 47 |
| struct 75% one-shot | -0.122 [-0.182, -0.062] | -0.564 [-0.637, -0.486] | +0.196 [+0.134, +0.247] | +0.546 [+0.388, +0.712] | - | accuracy, QWK, ECE, NLL | 220 | 48 | 97 |
| struct 75% +FT | -0.027 [-0.075, +0.020] | -0.007 [-0.087, +0.070] | -0.056 [-0.095, -0.003] | -0.150 [-0.260, -0.048] | ECE, NLL | - | 159 | 44 | 55 |
| struct 50% +FT +INT8 | -0.020 [-0.065, +0.022] | -0.017 [-0.101, +0.061] | +0.019 [-0.026, +0.057] | -0.037 [-0.147, +0.070] | - | - | 141 | 39 | 47 |

*Each cell is the change vs FP32 [95% bootstrap interval over the test images]. `changed` = test predictions that differ from FP32; `fixed` = wrong in FP32 and right now; `broken` = right in FP32 and wrong now.*

Recommended by validation score (tolerance 0.03): fastest = `struct 50% +FT +INT8`, smallest = `struct 50% +FT +INT8`.

![Latency](figures/latency.png)
![Speed against score](figures/tradeoff_latency_score.png)
![Sizes](figures/sizes.png)
![Pruning curves](figures/pruning_curves.png)

## 6. Key findings

- FP16 halved the file size (44.7 MB → 22.4 MB on both datasets) with no clear change in accuracy (0 of 624 predictions changed on PneumoniaMNIST, 1 of 400 on RetinaMNIST) but was slower on this CPU: 0.79× (95% CI 0.77 to 0.84) on PneumoniaMNIST and 0.95× (0.91 to 0.98) on RetinaMNIST.
- INT8 reduced the size to 11.3 MB and was 1.90× (CI 1.83 to 2.05) and 2.38× (CI 2.31 to 2.42) faster, but the accuracy change was −0.090 (CI −0.115 to −0.067; 58 predictions broken, 2 fixed) on PneumoniaMNIST and +0.007 (CI −0.018 to +0.037) on RetinaMNIST: the same recipe was clearly harmful on one dataset and showed no clear change on the other.
- Unstructured pruning left the file at 44.7 MB and gave no useful speed-up (0.88× to 0.94× on PneumoniaMNIST, 1.01× to 1.08× on RetinaMNIST), but at 90% the zip size fell from 41.5 to 7.9 MB; after fine-tuning, 90% pruning changed PneumoniaMNIST accuracy by −0.008 (CI −0.021 to +0.005), whereas the one-shot model had collapsed to accuracy 0.625.
- Structured pruning at 50% halved the file (22.7 MB) and gave a 2.23× (CI 1.84 to 2.40) and 2.29× (CI 2.22 to 2.32) speed-up; one-shot it destroyed the models (PneumoniaMNIST AUC 0.618, RetinaMNIST QWK 0.000), and fine-tuning recovered the main score to 0.985 and 0.550 (FP32: 0.985 and 0.564).
- The validation rule recommended `struct 75% +FT` as the fastest (11.7 MB, 3.99× faster, test AUC 0.974) and `struct 50% +FT +INT8` as the smallest model (5.8 MB, 3.88× faster, test AUC 0.980) for PneumoniaMNIST, and `struct 50% +FT +INT8` as both for RetinaMNIST (5.8 MB, 3.96× faster, test QWK 0.547 against 0.564 for FP32). On PneumoniaMNIST the smallest model nevertheless lost 0.071 test accuracy (CI −0.091 to −0.050), which a rule based on validation AUC does not check.

## 7. Limitations

- Speeds were measured on a **cloud CPU with 1 thread, not on a phone or clinic laptop**; only the ratios transfer, and only roughly. The machine was shared, so timings are noisy (the intervals are slightly optimistic).
- **One trained model per dataset** and small test sets (624 and 400 images): changes of a fraction of a percentage point are within the noise.
- Only post-training INT8 quantization with one calibration recipe; no quantization-aware training. FP16 was run on a CPU, where it usually has no native support.
- Unstructured pruning is simulated by zeros in a dense model (real gains would need sparse software or hardware). Structured pruning touched only the inner layer of each residual block.
- The fine-tuned variants received up to 8 more epochs of training than the FP32 model, and no unpruned fine-tuned control was run, so part of any gain after fine-tuning (for example the higher RetinaMNIST QWK at 90% unstructured pruning) may come from the extra training rather than from pruning.
- Memory use and energy were not measured. Low-resolution images. Not clinically validated.

## 8. How to reproduce

1. Click the **Open in Colab** badge and choose a GPU runtime (used only to fine-tune the pruned models; all speed tests run on the CPU).
2. Point `P1_DIR` in cell 1c to your Project 1 outputs (`reference_models.json` and `models/reference_*.pt`, created by running the Project 1 notebook).
3. Set `QUICK_TEST = False` and choose Runtime, Run all (about 15 to 20 minutes).

Exact library versions, the CPU model and all settings are in `environment.json`. Raw timing samples are in `latency_samples_*.npz`. To run locally: `pip install -r requirements_project3.txt`.

## 9. Repository contents

| File | What it is |
|---|---|
| `Project3_Model_Compression.ipynb` | Full code and explanations, with the outputs of my run |
| `results_tables.md`, `table1_*.csv`, `table2_changes_vs_fp32_*.csv` | Master table and changes vs FP32 |
| `latency_samples_*.npz` | Raw timing samples of the speed test |
| `figures/` | Plots used in this README |
| `environment.json` | Software versions, CPU model and settings |
| `requirements_project3.txt` | Python packages |

## 10. References and acknowledgements

- B. Jacob et al. *Quantization and training of neural networks for efficient integer-arithmetic-only inference.* CVPR, 2018.
- H. Li, A. Kadav, I. Durdanovic, H. Samet, H. P. Graf. *Pruning filters for efficient ConvNets.* ICLR, 2017.
- S. Han, J. Pool, J. Tran, W. J. Dally. *Learning both weights and connections for efficient neural networks.* NeurIPS, 2015.
- D. Blalock, J. J. Gonzalez Ortiz, J. Frankle, J. Guttag. *What is the state of neural network pruning?* MLSys, 2020 (benchmarking and reporting practice).
- P. Micikevicius et al. *Mixed precision training.* ICLR, 2018 (FP16).
- C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On calibration of modern neural networks.* ICML, 2017 (ECE).
- K. He, X. Zhang, S. Ren, J. Sun. *Deep residual learning for image recognition.* CVPR, 2016.
- J. Yang et al. *MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 2023.
- D. Kermany et al. *Identifying medical diagnoses and treatable diseases by image-based deep learning.* Cell, 2018 (source of PneumoniaMNIST).
- R. Liu et al. *DeepDRiD: Diabetic Retinopathy, Grading and Image Quality Estimation Challenge.* Patterns, 2022 (source of RetinaMNIST).
- ONNX Runtime documentation (quantization and performance tuning). Please also check the license terms of each dataset on the MedMNIST website.

## 11. Author

Nisha Pandey · [your email] · [link to your CV or LinkedIn]
