# Human Behavioral Face Recognition Data

This folder contains human behavioral data used to evaluate model–human similarity in face identity and emotion recognition tasks.

## Contents

- `human_trials.csv`  
  Pooled behavioral responses from all participants (2AFC task).

- `human_emotion_confusion.npy`
- `human_identity_confusion.npy`  
  Row-normalized human confusion matrices.

- `human_split_halves/`  
  Five independent split-half partitions (`A/B`) used to estimate behavioral reliability and noise ceilings.

## Task

Participants performed a two-alternative forced-choice (2AFC) match-to-sample task for:

- Face **identity** recognition
- Face **emotion** recognition

Each row corresponds to one behavioral trial.

## Purpose

These data provide human behavioral benchmarks for comparing computational face recognition models.
