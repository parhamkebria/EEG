# EEG benchmark datasets for the EEGIMAGE / IMAGINATION paper

Compiled 2026-08-25. Interactive version: https://claude.ai/code/artifact/58abcc97-2036-451f-8b62-8eb05c34bb64

> Companion to `eeg_benchmark_datasets.md` (the general curated list). This document is narrower and opinionated: it only covers datasets on which *other published methods already report numbers*, so the framework can be compared rather than re-benchmarked from scratch.

## Framing

The Berkeley Synchronized Brainwave set (single-channel MindWave, ~30 subjects, stimulus-condition labels) has no modern successor with a competitive literature, so there is nobody to compare against. EEGIMAGE is a generic trial-to-label encoder (features rendered as 2D arrays -> CNN), so the strategy is to enter three arenas that each carry a published leaderboard.

- **Arena A** — the 12-dataset suite used by EEG foundation-model papers (LaBraM ICLR'24, CBraMod ICLR'25, BIOT). Same splits, same metrics (balanced accuracy / Cohen's kappa / weighted F1), baselines already published.
- **Arena B** — MOABB motor-imagery benchmark. Maintained library, fixed evaluation protocol, ~30 ranked pipelines.
- **Arena C** — visual perception and mental imagery EEG. Justifies the IMAGINATION framing; newest and fastest-moving literature.

## Master list

| Dataset | Arena | Paradigm | Subj | Ch | Rate | Classes | Access |
|---|---|---|---|---|---|---|---|
| FACED | A | Emotion, 28 video clips | 123 | 32 | 250 Hz | 9 / 2 | Synapse syn50614194, CC BY 4.0 |
| SEED-V | A | Emotion + eye tracking | 16 | 62 | 1000 Hz | 5 | SJTU BCMI application |
| PhysioNet-MI (EEGMMIDB) | A, B | Motor imagery + execution | 109 | 64 | 160 Hz | 4 | PhysioNet, open |
| SHU-MI | A | Motor imagery L/R | 25 | 32 | 250 Hz | 2 | figshare, open |
| TUEV | A | Clinical EEG events | — | 16* | 250 Hz* | 6 | TUH signed form + rsync |
| TUAB | A | Normal vs abnormal | — | 16* | 250 Hz* | 2 | TUH signed form + rsync |
| Mumtaz2016 | A | MDD vs control | — | 19 | 256 Hz | 2 | figshare, open |
| MentalArithmetic (EEGMAT) | A | Workload vs rest | 36 | 20 | 500 Hz | 2 | PhysioNet, open |
| BNCI2014-001 (BCI IV-2a) | B | Motor imagery, 2 sessions | 9 | 22 | 250 Hz | 4 | MOABB, open |
| Cho2017 | B | Motor imagery L/R | 52 | 64 | 512 Hz | 2 | MOABB / GigaDB |
| Lee2019-MI (OpenBMI) | B | Motor imagery, 2 sessions, 11,000 trials | 54 | 62 | 1000 Hz | 2 | MOABB / GigaDB 10.5524/100542 |
| Dreyer2023 | B | Motor imagery + user profiles, ~240 trials each | 87 | 27 | 512 Hz | 2 | Zenodo 10.5281/zenodo.8089820 |
| THINGS-EEG2 (Gifford 2022) | C | Natural image RSVP; train 1654x10x4, test 200x1x80 | 10 | 63 | 1000 Hz | 200-way zero-shot | OSF 3jk45 |
| THINGS-EEG1 (Grootswagers 2022) | C | 1,854 concepts, 22,248 images, 10 Hz RSVP | 50 | 64 | 1000 Hz | 1854 | OpenNeuro ds003825, CC BY 4.0 |
| EEG-ImageNet (2024) | C | ImageNet-21k stimuli, 4,000 images, 63,850 pairs | 16 | 62 | 1000 Hz | 80 / 40 / 8 | github.com/Promise-Z5Q2SQ/EEG-ImageNet-Dataset |
| Visual Imagery BCI (Sci Data) | C | Imagined figures/animals/objects, 2 sessions, 4 s trials | 22 | 32 | 1000 Hz | 10 | figshare 10.6084/m9.figshare.30227503, CC BY-NC-ND |

\* TUH channel count/rate as harmonised by LaBraM/CBraMod preprocessing, not the raw clinical recordings.

Additional: Alljoined-1.6M (20 subj, 32-ch consumer, 1.6M trials, arXiv 2508.18571); MindBigData 2022 (MindWave 1ch / Insight 5ch / EPOC 14ch / Muse 4-5ch / Cap64, HuggingFace DavidVivancos/MindBigData2022); semantic concepts imagination-vs-perception (12 subj, 124 ch, 1024 Hz, OpenNeuro ds004306); HBN-EEG (3,000+ subjects, 6 tasks, 100 Hz, BIDS, s3://nmdatasets/NeurIPS2025/); ERP CORE (40 subj, 7 components, erpinfo.org/erp-core).

## Published baselines to beat (balanced accuracy)

| Dataset | EEGNet | LaBraM-Base | CBraMod |
|---|---|---|---|
| FACED (9-class) | 0.409 | 0.527 | 0.551 |
| SEED-V (5-class) | 0.296 | 0.398 | 0.409 |
| PhysioNet-MI (4-class) | 0.581 | 0.617 | 0.642 |
| SHU-MI (2-class) | 0.589 | — | 0.637 |

Others: FACED original paper 9-class cross-subject 35.2% (DE+SVM), 42.4% (CLISA). THINGS-EEG2 200-way zero-shot — BraVL 5.8/17.5, NICE 13.8/39.5 (top-1/top-5). EEG-ImageNet — 80-class 40.5% (RGNN), 40-class 53.4%, 8-class 81.6% (MLP). Visual Imagery BCI — EEGNet 75.8% animals, 75.1% figures, 62.0% objects. Dreyer2023 — mean online 63.35% +/- 17.36.

## Pitfalls

1. **Block-design leakage.** The 2017 Spampinato "EEG-ImageNet" 40-class dataset presented each class in a contiguous block; Li et al. (TPAMI 2021, *The Perils and Pitfalls of Block Design for EEG Classification Experiments*) and Ahmed & Wilbur (CVPR 2021) showed the 90%+ accuracies collapse to chance under randomised presentation. Do not use those numbers as baselines.
2. **EEG-ImageNet (2024) split.** Images grouped by category; official split = first 30 per category train, last 20 test, so temporally adjacent trials straddle the split. Report a temporally-blocked split alongside it.
3. **Subject-independent evaluation.** Keep the existing subject-aware split, state it as LOSO or a fixed subject-disjoint split, and report per-subject variance (Dreyer2023 ranges 40–99% on a binary task).
4. **Leakage through the rendering step.** Fit all normalisation, channel scaling and feature-image colour mapping on training subjects only.
5. **Metrics.** Use balanced accuracy + Cohen's kappa + weighted F1 (the LaBraM/CBraMod triple) — also the right answer to the repo's 31.6x class imbalance.

## Recommended plan

1. **Adopt the foundation-model protocol** — FACED, PhysioNet-MI, TUEV, TUAB under LaBraM/CBraMod preprocessing and metrics.
2. **Wrap EEGIMAGE as a MOABB pipeline** — BNCI2014-001, Cho2017, Lee2019-MI, Dreyer2023; splits and statistics come from the library.
3. **Earn the name** — Visual Imagery BCI (Sci Data) for true imagery + THINGS-EEG2 for perception/zero-shot.
4. **Close the loop on Berkeley** — MindBigData channel-count ablation (64 -> 1) turning the single-channel limitation into a stated contribution.

## Key sources

- CBraMod, ICLR 2025 — dataset suite + baseline tables
- LaBraM, ICLR 2024 — openreview QzTpTRVtrP
- NICE, ICLR 2024 — THINGS-EEG2 specs and zero-shot baselines
- FACED, Scientific Data 2023 — nature.com/articles/s41597-023-02650-w
- Dreyer et al., Scientific Data 2023 — nature.com/articles/s41597-023-02445-z
- MOABB dataset summary — moabb.neurotechx.com/docs/dataset_summary.html
- Li et al., TPAMI 2021 — block-design critique
- TUH EEG Corpus — isip.piconepress.com/projects/nedc/html/tuh_eeg/
