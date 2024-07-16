# Text-to-Speech Alignment Project

## Project Overview
This project explores and implements various text-to-speech (TTS) alignment techniques, aiming to improve the quality and efficiency of TTS systems. Our work spans multiple approaches, each addressing different aspects of the alignment challenge.

## Project Structure
This repository is organized into three main branches, each representing a distinct approach to TTS alignment:

1. [`MoBoAligner`](https://github.com/xiaozhah/Aligner/tree/MoBoAligner)
   - **Status**: Completed, for reference only
   - **Description**: Unofficial implementation of the "MoBoAligner: a Neural Alignment Model for Non-autoregressive TTS with Monotonic Boundary Search" paper
   - **Purpose**: Learning and baseline comparison
   - **Limitation**: Not suitable for large-scale applications due to maximum duration constraints

2. [`RoMoAligner`](https://github.com/xiaozhah/Aligner/tree/RoMoAligner)
   - **Status**: Development halted, for reference only
   - **Description**: Experimental improvement attempt combining Rough Alignment with MoBoAligner
   - **Purpose**: Explore self-supervised learning techniques in TTS alignment
   - **Limitation**: Performance improvements were limited and did not meet expectations

3. [`OTA`](https://github.com/xiaozhah/Aligner/tree/OTA) 👈 **Current Focus**
   - **Status**: In active planning and early development
   - **Description**: Adaptation of the "One TTS Alignment To Rule Them All" (OTA) method for implicit pause modeling
   - **Goal**: Develop a solution for handling implicit pauses without relying on explicit silence tokens
   - **Progress**: Conceptual development and planning phase

## Current Focus
Our primary focus is on the `OTA` branch, where we're exploring ways to adapt the OTA method for improved alignment, especially in handling implicit pauses in speech.

## How to Use This Repository
1. Check out each branch for specific implementation details and progress.
2. Refer to individual branch READMEs for setup and usage instructions.
3. For the latest developments, focus on the `OTA` branch.

## Contributing
We welcome contributions to any of our branches. If you're interested in contributing:
1. Check the issues in the relevant branch for tasks you can help with.
2. Fork the repository and create a pull request with your improvements.
3. For major changes, please open an issue first to discuss what you would like to change.

## Roadmap
- [x] Implement MoBoAligner (unofficial implementation)
- [x] Develop and test RoMoAligner
- [ ] Adapt and implement OTA for implicit pause modeling
- [ ] Conduct comparative studies across all methods
- [ ] Refine and optimize the most promising approach

## Acknowledgments
- Original [MoBoAligner paper](https://www.isca-speech.org/archive/interspeech_2020/li20h_interspeech.html)
- [OTA paper](https://arxiv.org/pdf/2108.10447)

We appreciate the support and interest from the TTS and speech processing community in advancing this research.