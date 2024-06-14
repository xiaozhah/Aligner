# RoMoAligner: Robust and Monotonic Alignment for Non-Autoregressive TTS

RoMoAligner is a novel alignment model designed for non-autoregressive Text-to-Speech (TTS) synthesis. It combines a rough aligner and a fine-grained monotonic boundary aligner (MoBoAligner) to achieve fast and accurate alignment between text and speech.

## Features

- Two-stage alignment: RoMoAligner first uses a rough aligner to estimate the coarse boundaries of each text token, then applies MoBoAligner to refine the alignment within the selected boundaries.
- Monotonic alignment: MoBoAligner ensures the monotonicity and continuity of the alignment, which is crucial for TTS.
- Robust and efficient: By selecting the most relevant mel frames for each text token, RoMoAligner reduces the computational complexity and improves the robustness of the alignment.
- Easy integration: RoMoAligner can be easily integrated into any non-autoregressive TTS system to provide accurate duration information.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/RoMoAligner.git
   cd RoMoAligner
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Compile the Cython extension:
   ```
   python setup.py build_ext --inplace
   ```

## Usage

```python
from romo_aligner import RoMoAligner

aligner = RoMoAligner(
    text_channels, mel_channels, attention_dim, attention_head, dropout, noise_scale
)

soft_alignment, hard_alignment, expanded_text_embeddings, dur_by_rough, dur_by_mobo = aligner(
    text_embeddings,
    mel_embeddings,
    text_mask,
    mel_mask,
    direction=["forward", "backward"],
)
```

## Model Architecture

RoMoAligner consists of two main components:

1. **RoughAligner**: A cross-modal attention-based module that estimates the coarse boundaries of each text token in the mel spectrogram.
2. **MoBoAligner (not official)**: A fine-grained monotonic boundary aligner that refines the alignment within the selected boundaries.

The rough aligner first provides an initial estimation of the text token durations, which are then used to select the most relevant mel frames for each token. MoBoAligner then performs a more precise alignment within these selected frames, ensuring the monotonicity and continuity of the alignment.

## Contributing

We welcome contributions to RoMoAligner! If you have any bug reports, feature requests, or suggestions, please open an issue on the [GitHub repository](https://github.com/yourusername/RoMoAligner/issues). If you'd like to contribute code, please fork the repository and submit a pull request.

## License

RoMoAligner is released under the [MIT License](LICENSE).

## Acknowledgements

We would like to thank the open-source community for their valuable contributions and feedback. Special thanks to the developers of [ESPnet](https://github.com/espnet/espnet) and [PyTorch](https://pytorch.org/) for their excellent libraries.
