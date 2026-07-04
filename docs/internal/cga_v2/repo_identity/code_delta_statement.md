# CGA-v2 Repository Identity Statement

This repository is a new repo-grade implementation.

Historical OHCM-MSHNet-main results are internal references only.
All paper evidence must be regenerated in this repository before being claimed.

This repo must pass model/loss/eval/data contract tests before any training result is accepted.

## Contract

- Model import contract.
- Model train/eval forward contract.
- Loss finite contract.
- CGA target generation contract.
- Eval final-logit-only contract.
- Threshold fixed at 0.5 unless explicitly changed in a frozen protocol.
- No test-time auxiliary head usage.
- No post-processing or verifier.
