# OCR-Sensitive Neurons

This repository contains the implementation and analysis for identifying OCR-sensitive neurons in neural networks used for named entity recognition (NER) tasks on historical documents. The goal is to improve the robustness of NER models against OCR errors commonly found in digitized historical texts.

This repository is the official implementation of the paper [**"Investigating OCR-Sensitive Neurons to Improve Entity Recognition in Historical Documents"**](https://arxiv.org/abs/2409.16934).

---

## Repository Structure

```
ocr-sensitive-neurons/
├── data/                # Example data for OCR and NER tasks
├── models/              # Pre-trained and fine-tuned models
├── notebooks/           # Jupyter notebooks for experiments
├── src/                 # Source scripts for model training and evaluation
├── scripts/             # Scripts for analysis and training
├── results/             # Output results and logs
├── requirements.txt     # Required Python packages
├── README.md            # Project documentation
└── LICENSE              # License information
```
---
## Introduction

Historical document digitization often introduces OCR errors that can hinder the performance of named entity recognition (NER) models. This project investigates OCR-sensitive neurons in transformer-based models and proposes methodologies to mitigate their impact.

### Key Goals:
- Identify neurons that are highly sensitive to OCR noise.
- Analyze the impact of OCR errors on NER performance.
- Propose strategies to improve model robustness against OCR errors.

---
The notebooks/ directory contains the following Jupyter notebooks for experiments and analysis:
1. Data.ipynb:
   - Prepares synthetic OCR-induced tokens for experiments.
   - Simulates OCR errors to understand their impact on NER tasks.
2. NoisyTextExperiments.ipynb:
   - Analyzes the prepared OCR tokens.
   - Passes tokens through LLaMA and Mistral models to identify OCR-sensitive regions and neurons.
3. Llama-MistralNEREvaluation.ipynb:
   - Evaluates the results of NER experiments.
   - Visualizes results, focusing on the effect of neutralizing OCR-sensitive neurons.

These notebooks provide an end-to-end exploration of OCR-sensitive neuron analysis, from data preparation to visualization of results.
---

The `src/` folder contains the primary scripts for running Named Entity Recognition (NER) experiments and analyzing the impact of OCR-sensitive neurons.

### Key Scripts:
1. **modeling_llama.py**:
   - Defines the model architecture and configuration for LLaMA.
   - Includes functions to neutralize OCR-sensitive neurons during inference.

2. **modeling_mistral.py**:
   - Similar to `modeling_llama.py` but tailored for Mistral.
   - Focuses on testing the effects of OCR-sensitive neurons on Mistral-based models.

3. **run_first_layers.sh**:
   - Shell script to run experiments with only the first few layers of the models active.
   - Helps identify OCR-sensitive regions in the early layers of the network.

4. **run_ner.py**:
   - Core script to execute NER tasks with and without neutralized neurons.
   - Provides metrics for evaluating the performance difference.

5. **run_test_layers25.sh**:
   - Similar to `run_first_layers.sh`, but focuses on testing layers 25 and beyond.
   - Used for deep analysis of OCR-sensitive regions in the later layers.

These scripts allow you to experiment with models, evaluate the effects of OCR-induced errors, and analyze how neutralizing neurons impacts NER accuracy and robustness.

---
