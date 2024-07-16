# Adapting OTA for Implicit Pause Modeling in TTS Alignment

## Overview

This project aims to adapt the "One TTS Alignment To Rule Them All" (OTA) method to implicitly model pauses and silences in Text-to-Speech (TTS) alignment, without relying on explicit silence tokens (sp) in the input text sequence.

## Background

- MoBoAligner had limitations in handling text duration.
- RoMoAligner, our previous novel approach, attempted to address these limitations through self-supervised learning but faced challenges in achieving satisfactory results.
- Current alignment methods often rely on explicit silence tokens (sp) which are not present in the raw text input to TTS systems.
- OTA method shows potential for flexible alignment but needs adaptation for implicit pause modeling.

## Research Goals

1. Modify OTA to implicitly model pauses and silences without explicit sp tokens in the input text.
2. Develop a flexible alignment system that can handle the discrepancy between text input (without pauses) and speech output (with natural pauses).
3. Improve TTS alignment quality by better handling non-explicit speech elements.

## Planned Approach

- Analyze how OTA can be adapted to infer pause positions without explicit tokens.
- Design modifications to OTA for implicit silence and pause modeling.
- Implement and test the adapted method using a Chinese dataset initially.
- Evaluate the method's effectiveness in capturing natural speech rhythms and pauses.

## Current Status

This project is in the planning phase. No code has been implemented yet.

## Future Work

- Implement the adapted OTA method.
- Test with datasets from multiple languages to ensure generalizability.
- Explore integration with various TTS systems to improve naturalness of synthesized speech.

## Contributing

We welcome input from researchers in speech processing, NLP, and TTS. Please open an issue for discussions or suggestions.

## Acknowledgments

Inspired by the original [OTA paper](https://arxiv.org/pdf/2108.10447).