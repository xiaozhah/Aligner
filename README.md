# RoMoAligner: An Experimental Approach Combining Rough Alignment with MoBoAligner

## Overview

RoMoAligner is an experimental implementation that combines a Rough Aligner with a [MoBoAligner](https://github.com/xiaozhah/Aligner/tree/MoBoAligner) to address limitations in text-to-speech alignment. This project is an original attempt to overcome the constraint of limited text duration (D) in the original MoBoAligner while maintaining its advantages.

## Key Features

- Combines Rough Aligner for initial boundary estimation with MoBoAligner for fine-grained alignment
- Attempts to remove the text duration (D) limitation of the original MoBoAligner
- Implements a novel self-supervised approach where fine alignments inform rough alignments

## Methodology

1. **Rough Alignment**: A rough aligner is used to obtain approximate text boundaries.
2. **Fine-grained Alignment**: MoBoAligner performs a more detailed search around these rough boundaries.
3. **Self-supervised Learning**: The fine-grained durations (dur_by_mobo) are used as a prediction target for the rough aligner's durations (dur_by_rough).

## Current Status

**Development Halted**: This implementation is currently not under active development due to challenges encountered during experimentation.

**Key Issue**: The loss between dur_by_mobo and dur_by_rough did not decrease as expected during training.

## Usage

## Usage

To use RoMoAligner in your project, follow these steps:

1. Import the RoMoAligner class:

```python
from romo_aligner import RoMoAligner
```

2. Initialize the RoMoAligner:

```python
aligner = RoMoAligner(
    text_embeddings=text_embeddings_dim,  # Dimension of text embeddings
    mel_embeddings=mel_embeddings_dim,    # Dimension of mel spectrogram embeddings
    attention_dim=128,                    # Dimension of attention
    attention_head=2,                     # Number of attention heads
    conformer_linear_units=256,           # Number of units in Conformer linear layers
    conformer_num_blocks=2,               # Number of Conformer blocks
    conformer_enc_kernel_size=7,          # Kernel size for encoder Conformer
    conformer_dec_kernel_size=31,         # Kernel size for decoder Conformer
    skip_text_conformer=False,            # Whether to skip text Conformer
    skip_mel_conformer=False,             # Whether to skip mel Conformer
    num_candidate_boundaries=3,           # Number of candidate boundaries
    verbose=True                          # Whether to print verbose information
)
```

3. Use the aligner to compute alignments:

```python
soft_alignment, hard_alignment, expanded_text_embeddings, dur_by_rough, dur_by_mobo = aligner(
    text_embeddings,     # Text embeddings (B, I, D_text)
    mel_embeddings,      # Mel spectrogram embeddings (B, J, D_mel)
    text_mask,           # Text mask (B, I)
    mel_mask,            # Mel mask (B, J)
    direction=["forward", "backward"]  # Alignment direction
)
```

The function returns:
- `soft_alignment`: Soft alignment matrix (B, I, J)
- `hard_alignment`: Hard alignment matrix (B, I, J)
- `expanded_text_embeddings`: Expanded text embeddings (B, J, D_text)
- `dur_by_rough`: Durations predicted by the rough aligner (B, I)
- `dur_by_mobo`: Durations searched by the MoBo aligner (B, I)

Note: Make sure to move the aligner to the same device as your input tensors (e.g., `aligner = aligner.to(device)`).

Remember to account for the experimental nature of this implementation when integrating it into your TTS pipeline.

## Limitations and Considerations

- This implementation is experimental and may not be suitable for production use.
- The self-supervised approach did not yield the expected improvements in alignment accuracy.
- Users should be aware that this version may not outperform the original MoBoAligner in all scenarios.

## Future Directions

While development has been paused, potential areas for future exploration include:
- Investigating alternative loss functions for the dur_by_mobo and dur_by_rough comparison
- Exploring different architectures for the rough and fine-grained aligners
- Considering hybrid approaches that leverage strengths from other alignment techniques

## Contributing

This project is currently not actively maintained. However, researchers and developers interested in the concept are encouraged to fork the repository and continue experimentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work is inspired by the original MoBoAligner [paper](https://www.isca-archive.org/interspeech_2020/li20ha_interspeech.html).

## Disclaimer

This is an experimental implementation based on original ideas extending the concept of MoBoAligner. It is provided "as-is" for research and educational purposes only. Users should exercise caution when considering this for any practical applications. While inspired by MoBoAligner, this implementation includes novel concepts not present in the original research.