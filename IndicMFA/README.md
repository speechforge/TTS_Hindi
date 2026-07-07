# IndicMFA Setup

This folder contains the dictionary and acoustic models for the Hindi Montreal Forced Aligner (MFA).

To properly use these models for alignment, please clone the official AI4Bharat IndicMFA repository directly into this folder and follow their setup instructions.

### 1. Clone the Repository
Run the following command inside this directory to clone the official repository:
```bash
git clone https://github.com/AI4Bharat/IndicMFA.git
```

### 2. Environment Setup
It is highly recommended to use Conda to install the Montreal Forced Aligner:
```bash
conda create -n aligner -c conda-forge montreal-forced-aligner
conda activate aligner
```