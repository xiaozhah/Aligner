# MoBoAligner: Unofficial Implementation

This branch contains an unofficial implementation of MoBoAligner (Monotonic Boundary Aligner) based on the paper "MoBoAligner: a Neural Alignment Model for Non-autoregressive TTS with Monotonic Boundary Search" by Li et al.

## Overview

MoBoAligner is a neural alignment model designed for non-autoregressive Text-to-Speech (TTS) systems. This unofficial implementation attempts to recreate the model described in the original paper, with some differences and limitations noted below.

## Key Features

- Monotonic boundary search for text-to-mel alignment
- End-to-end training framework
- Lightweight and task-specific design

## Important Limitation

This implementation, like the original MoBoAligner, has a significant limitation that restricts its use in large-scale TTS applications:

The computation of $\alpha_{i,j}$ and $\beta_{i,j}$ involves $P(B_i = j|B_{i-1} = k)$ and $P(B_i \geq j|B_{i-1} = k)$ respectively. In a standard implementation, $k$ would range from $0$ to $j-1$, and $j$ from $1$ to $J$, resulting in a tensor of shape $BIJJ$. For a 10-second audio clip with 10ms frames, $J$ would be 1000, making the $BIJJ$ tensor too large to fit in GPU memory.

To address this, MoBoAligner modifies the range of $k$ to be from $j-D$ to $j-1$, where $D$ is the **maximum duration** allowed for each text token. This modification allows the model to run, but it introduces a strict limitation on the maximum duration of each text token, potentially affecting the aligner's performance in TTS applications where longer token durations are necessary.

## Implementation Details

This unofficial implementation aims to follow the original paper, but there are some notable differences:

1. **Encourage Discreteness**: Our implementation uses a different approach to encourage discreteness in the alignment. Instead of using Gumbel-Softmax with annealing temperature as described in the paper, we employ a noise-based method to achieve a similar effect.

2. **Frame Interlacement**: The frame interlacement technique for acceleration, as mentioned in the paper, is not implemented in this version. This may result in longer training times compared to the paper's reported figures.

3. **Maximum Duration Limit**: As mentioned in the limitation section, this implementation enforces a maximum duration (D) for each text token. This is a critical difference from general-purpose aligners and may limit the model's applicability in certain TTS scenarios.

4. **Potential Differences**: As an unofficial implementation, there might be other subtle differences from the original work. Users should be aware that the performance and behavior might not exactly match those reported in the original paper.

## Usage

## Usage

To use MoBoAligner in your project, follow these steps:

1. Import the MoBoAligner class:

```python
from moboaligner import MoBoAligner
```

2. Initialize the MoBoAligner:

```python
aligner = MoBoAligner(
    text_channels,    # Dimension of text hidden states
    mel_channels,     # Dimension of mel spectrogram hidden states
    attention_dim     # Dimension of attention
)
```

3. Use the aligner to compute soft and hard alignments:

```python
soft_alignment, hard_alignment = aligner(
    text_hiddens,     # Text hidden states (B, I, D_text)
    mel_hiddens,      # Mel spectrogram hidden states (B, J, D_mel)
    text_mask,        # Text mask (B, I)
    mel_mask,         # Mel mask (B, J)
    direction=["forward", "backward"],  # Alignment direction
    return_hard_alignment=True  # Whether to return hard alignment
)
```

The `soft_alignment` is a tensor of shape (B, I, J) representing the soft alignment probabilities, while `hard_alignment` (if requested) is a binary tensor of the same shape representing the Viterbi path.

Note: Make sure to move the aligner to the same device as your input tensors (e.g., `aligner = aligner.to(device)`).

For integration with a TTS system, you can use MoBoAligner similarly to other alignment networks:

```python
self.aligner = MoBoAligner(
    text_channels=self.args.out_channels,
    mel_channels=self.args.n_hidden_conformer_encoder,
    attention_dim=128
)
```

Remember to account for the maximum duration limitation when using MoBoAligner in your TTS pipeline.

## Requirements

should be referenced from the `requirements.txt`.

## Disclaimer

This is an unofficial implementation and is not affiliated with or endorsed by the original authors. Results may vary from those reported in the original paper. Users should be aware of the limitations, particularly the maximum duration constraint, when considering this implementation for their projects.

## Citation

If you use this implementation in your research, please cite the original paper:

```
@inproceedings{li20ha_interspeech,
  author={Naihan Li and Shujie Liu and Yanqing Liu and Sheng Zhao and Ming Liu and Ming Zhou},
  title={{MoBoAligner: A Neural Alignment Model for Non-Autoregressive TTS with Monotonic Boundary Search}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={3999--4003},
  doi={10.21437/Interspeech.2020-1976},
  issn={2958-1796}
}
```

## Contributing

We welcome contributions to improve this unofficial implementation. Please feel free to submit issues or pull requests, especially if you have ideas on how to address the maximum duration limitation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Note: This is an unofficial implementation based on the research paper "MoBoAligner: a Neural Alignment Model for Non-autoregressive TTS with Monotonic Boundary Search" by Li et al. While this implementation is licensed under MIT, users should be aware of and respect any potential intellectual property rights associated with the original research.