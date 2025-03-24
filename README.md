
# Project Overview
This project involves the implementation of various cryptanalysis techniques, including Correlation Power Analysis (CPA) and Signal-to-Noise Ratio (SNR) analysis, using power traces. The provided code is designed to help analyze and break cryptographic algorithms by exploiting side-channel information.

## Installation and Setup
To run this project, you'll need to install the following Python libraries:
- **numpy**
- **matplotlib**
- **h5py**
- **palettable**

Use the following command to install the required libraries:
```bash
pip install numpy matplotlib h5py palettable
```

## Data Requirements
The project requires the `ascadv2_extracted.h5` dataset, which can be downloaded from the following link:
[Download Dataset](https://object.files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5)

Ensure that the dataset is placed in the same directory as the code files.

## Running the Code
### Running on Jupyter Notebook
1. Ensure you have Jupyter Notebook installed. If not, install it using:
   ```bash
   pip install notebook
   ```
2. Open the project folder in Jupyter Notebook.
3. Run the cells sequentially in each code file to execute the project.

### Running on MacOS
1. Open Terminal.
2. Navigate to the project directory.
3. Ensure you have Python installed (preferably Python 3.x).
4. Run the Python scripts:
   ```bash
   python filename.py
   ```

### Running on Windows
1. Open Command Prompt or PowerShell.
2. Navigate to the project directory.
3. Ensure you have Python installed (preferably Python 3.x).
4. Run the Python scripts:
   ```bash
   python filename.py
   ```

## File Descriptions

**Setup File**: Contains classes and functions used across the project:
- `multGF256(a, b)`: Multiplies two numbers in GF(256).
- `permIndices(i, m0, m1, m2, m3)`: Computes permutation indices.
- `RunningCorr`: Class for maintaining a running correlation.
- `RunningMean`: Class for maintaining a running mean.
- `RunningVar`: Class for maintaining a running variance.
- `CPA`: Implements Correlation Power Analysis.
- `CPA_2o`: Implements second-order Correlation Power Analysis.
- `RunningSNR`: Computes the running Signal-to-Noise Ratio (SNR).

**Tracing Mean**: Computes and plots the mean of power traces.

**SNR Analysis**:
- **Part 1**: Targets individual values for SNR analysis.
- **Part 2**: Plots all values together for comparison.

**First Order Correlation Power Analysis**: Implements first-order CPA and aligns power traces.

**Second Order Correlation Power Analysis**: Implements second-order CPA with enhanced trace alignment.

**Estimating Traces**: Estimates the number of traces required for a successful attack using a provided correlation coefficient.

## Example Instructions for Different Environments
Ensure that the dataset is properly loaded based on the environment:
- On **MacOS**, paths are typically case-sensitive.
- On **Windows**, use backslashes `\` in file paths.
- For **Jupyter Notebooks**, the dataset should be in the same directory as the notebook.

## Important Notes and Warnings
- **Dataset Placement**: Ensure the `ascadv2_extracted.h5` dataset is in the directory containing the code files.
- **Loading Data**: Check the data loading lines to ensure paths are correct depending on your operating system (MacOS, Windows).
- **Trace Alignment**: The alignment of traces might differ slightly based on the environment, so ensure the alignment functions are carefully adjusted.

## Acknowledgments
This project uses datasets and techniques developed in the field of cryptanalysis. Special thanks to the authors of the ascadv2 dataset and contributors to the libraries used in this project.
