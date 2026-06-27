# EEG Benchmark Datasets

A curated list of popular public EEG datasets used as benchmarks in research (BCI, motor imagery, emotion recognition, epilepsy, etc.). Maintained for easy reference in a Git repo.

## Motor Imagery / BCI

| Dataset | Subjects | Channels | Sampling Rate | Classes / Tasks | Description / Use Cases | Link |
|---------|----------|----------|---------------|-----------------|-------------------------|------|
| BCI Competition IV-2a | 9 | 22 EEG + 3 EOG | 250 Hz | 4 (left hand, right hand, feet, tongue) | Classic MI benchmark, two sessions | [Download](http://www.bbci.de/competition/iv/#dataset2a) / [Description](http://www.bbci.de/competition/iv/desc_2a.pdf) |
| BCI Competition IV-2b | 9 | 3 bipolar EEG + 3 EOG | 250 Hz | 2 (left/right hand) | MI with feedback in later sessions | [Download](http://www.bbci.de/competition/iv/#dataset2b) |
| PhysioNet EEG Motor Movement/Imagery (EEGMMIDB) | 109 | 64 | ~160 Hz | Motor execution & imagery (fists, feet) + baselines | Large-scale MI and movement dataset | [PhysioNet](https://www.physionet.org/content/eegmmidb/1.0.0/) |
| BCI Competition IV-1 | 7 | 64 | 1000 Hz | 2-3 classes + idle | Self-paced MI | [Download](http://www.bbci.de/competition/iv/#dataset1) |
| High-Gamma Dataset | 14 | 128 | - | 4 (left/right hand, feet, rest) | High-frequency MI | [GitHub](https://github.com/robintibor/high-gamma-dataset) |

## Emotion Recognition

| Dataset | Subjects | Channels | Sampling Rate | Tasks | Description | Link |
|---------|----------|----------|---------------|-------|-------------|------|
| DEAP | 32 | 32 | 512 Hz | Emotion elicitation via music videos | Valence/arousal ratings + physiological | [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) |
| SEED | 15 | 62 | - | Positive/negative/neutral videos | Multiple sessions | [SEED](http://bcmi.sjtu.edu.cn/~seed/seed.html) |
| SEED-IV | 15 | 62 | - | Happy/sad/neutral/fear | With eye-tracking | [SEED-IV](http://bcmi.sjtu.edu.cn/~seed/seed-iv.html) |

## Epilepsy / Clinical

| Dataset | Subjects | Duration | Description | Link |
|---------|----------|----------|-------------|------|
| CHB-MIT Scalp EEG | 24 (pediatric) | ~982 hours | Seizure detection benchmark | [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) |
| Siena Scalp EEG | 14 | ~128 hours | Epilepsy recordings | [PhysioNet](https://physionet.org/content/siena-scalp-eeg/1.0.0/) |

## Other Notable Datasets

| Dataset | Category | Subjects | Key Features | Link |
|---------|----------|----------|--------------|------|
| Grasp and Lift EEG Challenge | Movement | 12 | 32 channels, grasp/lift events | [Kaggle](https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data) |
| bigP3BCI | P300 BCI | Varies | P300 speller, diverse conditions | [PhysioNet](https://physionet.org/content/bigp3bci/) |
| EEG-ImageNet | Visual | 16 | Image viewing for classification/reconstruction | [arXiv](https://arxiv.org/abs/2406.07151) |

## Resources & Repositories
- [Comprehensive EEG Datasets List (GitHub)](https://github.com/meagmohit/EEG-Datasets) — Highly recommended starting point.
- [PhysioNet EEG Databases](https://www.physionet.org/about/database/)
- [OpenNeuro EEG Search](https://openneuro.org/search/modality/eeg)
- [BNCI Horizon 2020 Datasets](https://bnci-horizon-2020.eu/database/data-sets)
- [MOABB / TorchEEG](https://torcheeg.readthedocs.io/) — Toolboxes with many pre-loaded datasets.

**Notes**:
- Always cite original papers when using these datasets.
- Check licenses and data formats (EDF, .mat, etc.).
- Preprocessing standards vary; many papers use MNE-Python or similar.
- Last updated: June 2026. Contributions welcome via PRs!

This table can be expanded as needed.