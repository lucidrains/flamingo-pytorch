<img src="./flamingo.png" width="500px"></img>

## Flamingo - Pytorch (wip)

Implementation of <a href="https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model">Flamingo</a>, state-of-the-art few-shot visual question answering attention net, in Pytorch. It will include the perceiver resampler, the specialized cross attention block (that includes the learned queries concatted key / values in addition to what is being cross attended, as well as block masking), as well as the interesting tanh gating at the ends of the cross attention + corresponding feedforward blocks

## Citations

<a href="https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf">Preprint</a>
