# ELEC5305 Project — Speaker Anonymisation

## Project Overview

This project investigates speaker anonymisation using both classical
signal processing and modern deep learning approaches.

The project will compare an LPC/McAdams-based classical anonymisation
method with zero-shot neural voice conversion using Seed-VC.

The main objective is to investigate the privacy–utility trade-off:
how effectively speaker identity can be concealed while preserving
speech intelligibility.

## Research Question

**How does zero-shot neural voice conversion compare with classical
LPC-based anonymisation in protecting speaker identity while preserving
speech intelligibility?**

## Planned Methods

### Classical Signal Processing
- MATLAB
- Linear Predictive Coding (LPC)
- Source-filter modelling
- McAdams-based spectral modification

### Deep Learning
- Python / PyTorch
- Seed-VC zero-shot voice conversion
- Neural speaker anonymisation

### Evaluation
- Speaker verification / speaker similarity for privacy evaluation
- Automatic speech recognition and Word Error Rate (WER)
- Acoustic analysis

## Project Structure

- `matlab/` — classical LPC/McAdams anonymisation
- `python/` — Seed-VC neural anonymisation and evaluation
- `data/` — dataset information and preparation
- `samples/` — selected audio examples
- `results/` — experimental results and figures
