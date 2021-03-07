# Expressive Tacotron (implementation with Pytorch)

This repository provides a multi-speaker expressive speech synthesis framework. The framework includes various deep learning architectures such as **Global Style Token (GST)**, **Variational Autoencoder (VAE)**, and **Gaussian Mixture Variational Autoencoder (GMVAE)**, and **X-vectors** for building prosody encoder.

This repository also provides a multi-mode Tacotron framework, including **multi-attentive Tacotron**, **DurIAN**, **Non-attentive Tacotron**.


## Available recipes

### Expressive Mode
- [x] [Global Style Token GST](https://arxiv.org/abs/1803.09017)
- [x] [Variational Autoencoder VAE](https://arxiv.org/abs/1812.04342)
- [x] [Gaussian Mixture VAE GMVAE](https://arxiv.org/abs/1810.07217)
- [x] X-vectors

### Framework Mode
- [x] [Tacotron2](Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions.)
- [x] [ForwardAttention](https://arxiv.org/abs/1807.06736)
- [x] [DurIAN](https://arxiv.org/abs/1909.01700)
- [x] [Non-attentive Tacotron](https://arxiv.org/abs/2010.04301)
- [ ] [GMM Attention](https://arxiv.org/pdf/1910.10288.pdf) (Todo)
- [ ] [Dynamic Convolution Attention](https://arxiv.org/pdf/1910.10288.pdf) (Todo)


## Differences
- Only provides **kernel model files**, not including **data prepared scripts**, **training scripts** and **synthesis scripts**
- ForwardAttention: based on LSA
- DurIAN: CBHG Encoder is replaced with Tacotron2 Encoder, also supports Skipped Encoder
- Non-attentive Tacotron: duration stacked convolution layers are concatenated with encoder outputs
- Default PostNet: CBHG


### Acknowledgements
This implementation uses code from the following repos: [NVIDIA](https://github.com/NVIDIA/tacotron2), [ESPNet](https://github.com/espnet/espnet), [ERISHA](https://github.com/ajinkyakulkarni14/ERISHA), [ForwardAttention](https://github.com/jxzhanggg/nonparaSeq2seqVC_code/blob/master/pre-train/model/basic_layers.py)

