<img src="./flamingo.png" width="500px"></img>

## Flamingo - Pytorch

Implementation of <a href="https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model">Flamingo</a>, state-of-the-art few-shot visual question answering attention net, in Pytorch. It will include the perceiver resampler, the specialized cross attention block (that includes the learned queries concatted key / values in addition to what is being cross attended, as well as block masking), as well as the interesting tanh gating at the ends of the cross attention + corresponding feedforward blocks

## Install

```bash
$ pip install flamingo-pytorch
```

## Usage

```python
import torch
from flamingo_pytorch import PerceiverResampler

perceive = PerceiverResampler(
    dim = 1024,
    depth = 2,
    dim_head = 64,
    heads = 8,
    num_latents = 64,    # the number of latents to shrink your media sequence to, perceiver style
    num_time_embeds = 4  # say you have 4 images maximum in your dialogue
)

medias = torch.randn(1, 2, 256, 1024) # (batch, time, sequence length, dimension)
resampled = perceive(medias) # (1, 2, 64, 1024) - (batch, time, num latents, dimension)
```

Then you insert the `GatedCrossAttentionBlock` at different intervals in your giant language model. Your text would then attend to the perceived media from above


```python
import torch
from flamingo_pytorch import GatedCrossAttentionBlock

cross_attn = GatedCrossAttentionBlock(
    dim = 1024,
    dim_head = 64,
    heads = 8
)

text = torch.randn(1, 512, 1024)
perceived_media = torch.randn(1, 2, 64, 1024)

time_mask = torch.ones((1, 2)).bool()
media_locations = torch.randint(0, 2, (1, 512)).bool()

text = cross_attn(
    text,
    perceived_media,
    time_mask = time_mask,
    media_locations = media_locations
)
```

That's it!

Attention is all you need.

## Todo

- [ ] integrate with  https://github.com/lucidrains/vit-pytorch and https://github.com/lucidrains/palm-pytorch , do full end-to-end example, and perhaps take care of the function that inserts every so often in the backbone of the LLM

## Citations

```bibtex
@article{Alayrac2022Flamingo,
  title   = {Flamingo: a Visual Language Model for Few-Shot Learning},
  author  = {Jean-Baptiste Alayrac et al},
  year    = {2022}
}
```
