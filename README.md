# ECG Anomaly Detection — Autoencoder Neural Network

Detecting anomalous ECG heartbeat signals from normal ones 
using a custom Autoencoder neural network built with 
TensorFlow and Keras — trained exclusively on normal data 
and evaluated by reconstruction error thresholding.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)
![Task](https://img.shields.io/badge/Task-Anomaly%20Detection-red)
![Method](https://img.shields.io/badge/Method-Autoencoder-green)

---

## Overview

Implements a deep Autoencoder architecture using a custom 
TensorFlow `Model` subclass to detect anomalous ECG 
(electrocardiogram) heartbeat signals. The Autoencoder is 
trained **only on normal ECG data** — it learns to 
reconstruct normal heartbeat patterns. When presented with 
anomalous signals, reconstruction error is significantly 
higher, enabling anomaly detection via a threshold computed 
from the training loss distribution. The pipeline covers 
data loading, MinMax normalisation, normal/anomaly 
separation, encoder-decoder architecture design, early 
stopping training, reconstruction visualisation, and 
loss histogram thresholding.

---

## Problem Statement

ECG anomaly detection is a critical medical application — 
identifying irregular heartbeat patterns early can prevent 
life-threatening cardiac events. Traditional supervised 
classification requires large amounts of labelled anomaly 
data, which is often scarce in medical settings. 
Autoencoders offer a powerful unsupervised alternative: 
train only on normal data, and flag any signal the model 
cannot reconstruct well as anomalous.

---

## Dataset

- **Name:** ECG5000 — ECG Heartbeat Dataset
- **Source:** Loaded directly from TensorFlow's public 
  storage — no manual download needed:
```python
  df = pd.read_csv(
      'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv',
      header=None
  )
```
- **Size:** 5,000 ECG signal records
- **Shape:** (5000, 141) — 140 time-step features 
  + 1 label column
- **Label column (c0):**
  - `0` = Normal heartbeat
  - `> 0` = Anomalous heartbeat (various cardiac 
    abnormality types)
- **Format:** CSV — each row is one complete ECG 
  heartbeat signal as a time series of amplitude values

> **No download required.** The dataset is fetched 
> automatically from the URL above when the notebook runs.

---

## Approach

### Data Preprocessing
- Loaded ECG dataset directly from TensorFlow's URL 
  using `pd.read_csv()`
- Added column prefixes using `.add_prefix('c')` 
  for clarity
- Split into 80% training and 20% test sets using 
  `train_test_split(random_state=111)`
- Applied **MinMaxScaler** fitted on training data only; 
  transformed both train and test sets separately 
  to prevent data leakage

### Normal / Anomaly Separation
- Separated scaled data into four subsets:
  - `normal_train_data` — label `c0 == 0` (training)
  - `anomaly_train_data` — label `c0 > 0` (training)
  - `normal_test_data` — label `c0 == 0` (test)
  - `anomaly_test_data` — label `c0 > 0` (test)
- Visualised sample normal ECG signals (blue) and 
  anomalous ECG signals (blue) to confirm visible 
  differences in waveform shape

### Autoencoder Architecture

A custom `AutoEncoder` class subclassing 
`tf.keras.Model` was built with separate 
`encoder` and `decoder` sub-networks:

**Encoder — compresses input to bottleneck:**

| Layer | Units | Activation |
|-------|-------|------------|
| Dense | 64 | ReLU |
| Dense | 32 | ReLU |
| Dense | 16 | ReLU |
| Dense (bottleneck) | 8 | ReLU |

**Decoder — reconstructs from bottleneck:**

| Layer | Units | Activation |
|-------|-------|------------|
| Dense | 16 | ReLU |
| Dense | 32 | ReLU |
| Dense | 64 | ReLU |
| Dense (output) | 140 | ReLU |

```python
class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8,  activation='relu')   # bottleneck
        ])
        self.decoder = tf.keras.Sequential([
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(140, activation='relu')  # reconstructed output
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Input | Normal ECG signals only |
| Target | Same as input (reconstruction) |
| Loss Function | MAE (Mean Absolute Error) |
| Optimiser | Adam |
| Epochs | 50 (max) |
| Batch Size | 120 |
| Early Stopping | patience=2, monitor=val_loss |

> The Autoencoder is trained **only on normal data** — 
> it never sees anomalous signals during training. 
> This is the key principle of unsupervised 
> anomaly detection.

### Reconstruction Evaluation

**Visual comparison:**
- Plotted original signal (blue) vs reconstructed 
  signal (red) for:
  - Normal test data — reconstruction should be close
  - Anomaly test data — reconstruction should deviate 
    significantly

**Threshold-based anomaly detection:**
```python
threshold = np.mean(train_loss) + 2 * np.std(train_loss)
```
- Computed MAE reconstruction loss for both 
  normal and anomaly test sets
- Plotted overlapping histograms of normal loss 
  vs anomaly loss with the threshold as a vertical 
  dashed red line
- Signals with reconstruction loss above the 
  threshold are classified as anomalous


## Visualisations

The notebook produces five plots:

1. **Normal ECG signals** — 3 sample normal 
   heartbeat waveforms overlaid
2. **Anomaly ECG signals** — 3 sample anomalous 
   heartbeat waveforms overlaid
3. **Normal reconstruction** — original vs 
   reconstructed normal signal (blue vs red)
4. **Anomaly reconstruction** — original vs 
   reconstructed anomaly signal (blue vs red)
5. **Loss histogram** — overlapping histograms of 
   normal and anomaly MAE reconstruction losses 
   with dashed threshold line

---

## Technologies Used

- **Language:** Python 3
- **Deep Learning:** TensorFlow, Keras 
  (Model subclassing, Sequential, Dense, 
  EarlyStopping callback)
- **Preprocessing:** Scikit-Learn 
  (MinMaxScaler, train_test_split)
- **Data Handling:** Pandas, NumPy
- **Visualisation:** Matplotlib

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/OyelolaIbrahim/ecg-anomaly-detection-autoencoder.git
cd ecg-anomaly-detection-autoencoder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
#    Dataset loads automatically from the internet
jupyter notebook autoencoder_ecg.ipynb
```
