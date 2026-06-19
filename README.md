# biiigProject: CPP Latent Dynamics from Single-Trial EEG

## 1. Project Overview

This project studies whether a neural network can learn useful latent representations from single-trial EEG signals related to centro-parietal positivity (CPP), and whether those learned representations are meaningfully associated with behavioural response time.

The repository is organized as a complete course-style data analysis project. It includes:

- project documentation
- processed data for analysis
- Python code for preprocessing, modeling, validation, and behavioural analysis
- saved figures and result tables
- tests for core pipeline components

The main workflow uses a causal forward GRU model trained on response-locked EEG from the CPP-related channels `CP1`, `CP2`, and `CPz`.

## 2. Research Question

This project addresses two linked questions:

1. Can a neural network reconstruct and preserve key CPP-related EEG structure from single-trial response-locked signals?
2. Do the learned hidden representations contain useful information about behavioural response time beyond simple baseline covariates?

The project is not a paper replication assignment. It is presented as an original analysis project built around EEG latent dynamics and behaviour.

## 3. Project Structure

The repository uses a course-friendly folder layout so that data, code, results, and tests are easy to locate.

```text
biiigProject/
├── README.md                         # Main project report and usage guide
├── requirements.txt                  # Python dependencies with pinned versions
├── AGENTS.md                         # Project routing and workspace rules
├── logs.md                           # Dated experiment and decision log
├── conftest.py                       # Root pytest path bootstrap
├── pytest.ini                        # Test configuration
├── Data/
│   ├── InputData/
│   │   └── Metadata/
│   │       └── DataSourcesGuide.md   # Data source and file reference notes
│   ├── ProcessedData/                # Main analysis-ready EEG and metadata files
│   └── IntermediateData/
│       └── latents_full/             # Exported hidden-state representations
├── Scripts/
│   ├── master_pipeline.ipynb         # Recommended main entry point
│   ├── pipeline_overview.md          # Written stage-by-stage pipeline description
│   ├── s0_preprocessing/             # Dataset construction and preprocessing scripts
│   ├── s1_modeling/                  # Model, dataset loading, and data checks
│   ├── s2_training/                  # Training, controls, sweeps, and CLI entry points
│   ├── s3_validation/                # Validation routing notes
│   └── s4_analysis/                  # Behavioural analysis, figures, and analysis notebooks
│       └── notebooks/                # Step-by-step exploratory analysis notebooks
├── Results/
│   ├── model_checkpoints/            # Saved trained model checkpoint
│   ├── validation/                   # Validation tables, reports, and figures
│   ├── regression/                   # Behavioural regression outputs
│   └── figures/                      # Saved diagnostic figures
└── tests/                            # Automated tests for core pipeline components
```

In short:

- `Data/` stores analysis inputs and intermediate representations
- `Scripts/` stores the main code and notebook workflow
- `Results/` stores output figures, tables, and trained-model artifacts
- `tests/` stores automated checks for the core pipeline
- `README.md` is the main document a reviewer can use to understand the project

The repository was previously reorganized from an earlier transition layout. The active materials are now consolidated into the root-level `Data/`, `Scripts/`, `Results/`, and `tests/` structure shown above.

## 4. Data Description

### 4.1 Data Provided in This Repository

This repository mainly provides processed data and generated results rather than the full original raw export. The included analysis-ready data are sufficient to inspect the workflow, understand the variables, and reproduce the main analysis steps once the necessary larger files are available.

Main processed dataset:

- `Data/ProcessedData/eeg_cpp_trials.npy`
- `Data/ProcessedData/times_ms.npy`
- `Data/ProcessedData/metadata.csv`
- `Data/ProcessedData/channel_names.txt`
- `Data/ProcessedData/preprocessing_notes.md`

Main intermediate representation:

- `Data/IntermediateData/latents_full/latents_full.npz`

### 4.2 Dataset Size

The current processed dataset contains:

- `7297` retained trials
- `308` time points per trial
- `3` EEG channels: `CP1`, `CP2`, `CPz`
- `41` participants

Array shape of the main EEG file:

- `eeg_cpp_trials.npy`: `(7297, 308, 3)`

Array shape of the latent representation:

- `latents_full.npz["latents"]`: `(7297, 308, 32)`

### 4.3 Main Variables

The trial-level metadata file contains behavioural and trial descriptors, including:

- `subject_id`
- `trial_id`
- `RT_ms`
- `correctness`
- `condition`
- `difficulty`
- `evidence_strength`
- `choice`
- `response_hand`
- `alignment`

The metadata table currently has `7297` rows and `28` columns.

### 4.4 Data Quality and Missingness

The processed dataset was built after removing trials with non-finite EEG values. According to the preprocessing notes:

- original trial pairs before removal: `10258`
- removed trials with invalid EEG values: `2961`
- retained trials for analysis: `7297`

The current metadata file does not show missing values in the main analysis columns used in the project summary.

### 4.5 Data Access

The full data package was not uploaded to GitHub because the files are too large for convenient repository storage. This repository therefore focuses on the processed analysis structure, saved outputs, and reproducible code organization.

If full reproduction of the larger data-dependent steps is needed, the complete data can be requested by contacting the author through GitHub.

## 5. Methods and Workflow

The analysis pipeline follows four main stages.

### 5.1 Data Preparation

The project first organizes response-locked EEG and behavioural metadata into a trial-level dataset. The focus is on the CPP-related channels `CP1`, `CP2`, and `CPz`.

Key outputs of this stage:

- trial-by-time EEG array
- time axis in milliseconds
- cleaned behavioural metadata
- channel-name record

### 5.2 Model Training

A causal forward GRU is trained on the EEG data. The model is designed to learn hidden representations that preserve important temporal structure while reconstructing the signal and predicting the immediate future of the waveform.

Training emphasizes:

- current-frame reconstruction
- short-horizon future prediction
- preservation of CPP-related waveform structure

### 5.3 Neural Validation

After training, the saved hidden states are checked against EEG-based targets. This stage asks whether the learned representation still carries meaningful CPP-related information rather than only producing visually plausible outputs.

This validation includes:

- neural reconstruction fit
- CPP waveform comparison
- hidden-state to CPP prediction checks
- control comparisons against weaker baselines

### 5.4 Behavioural Analysis

The final stage examines whether the learned hidden states help explain behavioural response time.

This stage includes:

- window-based hidden-state summarization
- ridge regression for `log(RT_ms)`
- baseline-versus-hidden comparison
- saved plots and result tables

## 6. Main Results

### 6.1 Core Results

The strongest results support the two main claims of the project.

#### A. The model captures key CPP-related EEG structure well

Main neural validation results:

- held-out neural reconstruction `R^2 = 0.8810`
- empirical-predicted neural correlation `= 0.9407`
- CPP waveform correlation `= 0.9981`
- CPP amplitude prediction in the broad pre-response window reached very high agreement in validation summaries

These results support the claim that the model learned a useful representation of the response-locked EEG structure rather than only fitting noise.

#### B. Hidden representations carry behaviour-related information

The behavioural regression results show that hidden-state features improve response-time prediction in the best pre-response window:

- baseline `R^2 = 0.197`
- baseline plus hidden states `R^2 = 0.300`
- improvement `delta R^2 = 0.103`

This supports the project’s second main conclusion: the learned latent representation contains information related to behavioural response time.

### 6.2 Supplementary Results

Additional checks help define what the model does and does not capture well.

- hidden states strongly preserve CPP amplitude information
- RT-related decoding is more stable than condition or choice decoding
- choice and condition decoding are close to chance in the reported summaries
- some auxiliary task-variable findings are weak and should not be overinterpreted

These supplementary results strengthen the interpretation that the model is more successful at preserving CPP-related and broad RT-related structure than at capturing fine-grained task identity.

## 7. How to Run

### 7.1 Environment

Recommended environment:

- Python `3.13`
- Jupyter Notebook

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

### 7.2 Main Entry Point

The recommended starting point is:

```bash
jupyter notebook Scripts/master_pipeline.ipynb
```

This notebook is the clearest reviewer-facing entry point because it walks through the pipeline step by step.

### 7.3 Shortest Reviewer Path

For a reviewer or course instructor opening the repository for the first time:

1. Open `README.md`
2. Open `Scripts/master_pipeline.ipynb`
3. Inspect the saved outputs in `Results/validation/` and `Results/regression/`
4. Run the tests if needed

Test command:

```bash
python -m pytest tests/ -v
```

### 7.4 Command-Line Alternatives

Key command-line entry points also exist through:

```bash
python Scripts/s2_training/cli.py validate --dataset-dir Data/ProcessedData
python Scripts/s2_training/cli.py train --dataset-dir Data/ProcessedData --output-dir Results/model_checkpoints
python Scripts/s2_training/cli.py extract-latents --dataset-dir Data/ProcessedData --checkpoint-path Results/model_checkpoints/best_model.pt --output-dir Data/IntermediateData/latents_full
python Scripts/s2_training/cli.py ridge-rt --dataset-dir Data/ProcessedData --latent-path Data/IntermediateData/latents_full/latents_full.npz --output-dir Results/regression
```

### 7.5 What Can Be Viewed Without the Full Large Data Package

Even without the full larger source-data package, a reviewer can still:

- inspect the project structure
- read the pipeline notebook
- view saved result tables and figures
- inspect processed metadata and documentation
- run tests on the core modeling and analysis code

For complete end-to-end regeneration of all large data-dependent steps, the full data package is still required.

## 8. Reproducibility and Data Access

This repository was prepared to show a complete project structure with documentation, code, data descriptions, saved outputs, and tests.

Reproducibility status:

- core project code is included
- saved outputs and figures are included
- processed data description is included
- tests for major pipeline components are included
- full large raw/working data are not fully bundled in GitHub due to size limits

If full reproduction is required for review, the author can provide the larger data files upon request through GitHub contact.

For additional file-level details, see:

- `Scripts/pipeline_overview.md`
- `Data/InputData/Metadata/DataSourcesGuide.md`
- `Data/ProcessedData/preprocessing_notes.md`

## 9. Limitations

This project has several important limitations that should be stated clearly.

- The full large data package is not included in the GitHub repository because of file size constraints.
- The repository is strongest as a structured analysis submission with saved outputs, processed data description, and runnable code, rather than as a fully self-contained large-data archive.
- Some auxiliary decoding results are weak or near chance, especially for choice and condition.
- The project supports cautious claims about CPP-related neural structure and response-time association, not overly strong claims about all task variables.
- One originally listed direction, DDM regression, is not yet completed in the current README status summary.

## 10. References

- Kosciessa, J. Q., et al. (2021). *Thalamocortical excitability modulation guides human perception under uncertainty*. Nature Communications, 12, 2430. https://doi.org/10.1038/s41467-021-22511-7
- van Bergen, R. S., & Jehee, J. F. M. (2019). Reference behavioural/CPP material used for audit comparison in repository notes.
- TIER Protocol 4.0. [https://www.projecttier.org/tier-protocol/protocol-4-0/](https://www.projecttier.org/tier-protocol/protocol-4-0/)
