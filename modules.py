# modules.py includes various encoders, GST, VAE, GMVAE, X-vectors
# adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
# MIT License
#
# Copyright (c) 2018 MagicGirl Sakura
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from TDNN_gpu import TDNN, TDNN_cpu, StatsPooling, FullyConnected

from gmvae import GMVAENet


class SpeakerEncoderNetwork(nn.Module):
    def __init__(self, hp):
        super().__init__()
        if hp.speaker_encoder_type.lower() == 'gst':
            self.encoder = GST(hp, hp.speaker_embedding_size, hp.speaker_classes)
        elif hp.speaker_encoder_type.lower() == 'vae':
            self.encoder = VAE(hp, hp.speaker_embedding_size, hp.speaker_classes)
        elif hp.speaker_encoder_type.lower() == 'gst_vae':
            self.encoder = GST_VAE(hp, hp.speaker_embedding_size, hp.speaker_classes)
        elif hp.speaker_encoder_type.lower() == 'gmvae':
            self.encoder = GMVAE(hp, hp.speaker_embedding_size, hp.speaker_classes)
        elif hp.speaker_encoder_type.lower() == 'x-vector':
            self.encoder = X_vector(hp, hp.speaker_embedding_size, hp.speaker_classes)
        elif hp.speaker_encoder_type.lower() == 'vqvae':
            raise ValueError("Error: unsupported type of 'vqvae'")
        else:
            raise ValueError("Erroe: unsupported type of 'speaker encoder'")

    def forward(self, inputs, input_lengths=None):

        embedding, cat_prob = self.encoder(inputs, input_lengths)

        return (embedding, cat_prob)


class ExpressiveEncoderNetwork(nn.Module):
    def __init__(self, hp):
        super().__init__()
        if hp.expressive_encoder_type.lower() == 'gst':
            self.encoder = GST(hp, hp.token_embedding_size, hp.emotion_classes)
        elif hp.expressive_encoder_type.lower() == 'vae':
            self.encoder = VAE(hp, hp.token_embedding_size, hp.emotion_classes)
        elif hp.expressive_encoder_type.lower() == 'gst_vae':
            self.encoder = GST_VAE(hp, hp.token_embedding_size, hp.emotion_classes)
        elif hp.expressive_encoder_type.lower() == 'gmvae':
            self.encoder = GMVAE(hp, hp.token_embedding_size, hp.emotion_classes)
        elif hp.expressive_encoder_type.lower() == 'x-vector':
            self.encoder = X_vector(hp, hp.token_embedding_size, hp.emotion_classes)
        elif hp.expressive_encoder_type.lower() == 'vqvae':
            raise ValueError("Error: unsupported type of 'vqvae'")
        else:
            raise ValueError("Erroe: unsupported type of 'speaker encoder'")

    def forward(self, inputs, input_lengths=None):

        embedding, cat_prob = self.encoder(inputs, input_lengths)

        return (embedding, cat_prob)


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, hp):

        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(hp.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = hp.n_mel_channels
        self.ref_enc_gru_size = hp.ref_enc_gru_size

    def forward(self, inputs, input_lengths=None):
        assert inputs.size(-1) == self.n_mel_channels
        out = inputs.unsqueeze(1)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        if input_lengths is not None:
            # print(input_lengths.cpu().numpy(), 2, len(self.convs))
            input_lengths = (input_lengths.cpu().numpy() / 2 ** len(self.convs))
            input_lengths = max(input_lengths.round().astype(int), [1])
            # print(input_lengths, 'input lengths')
            out = nn.utils.rnn.pack_padded_sequence(
                out, input_lengths, batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, l, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            l = (l - kernel_size + 2 * pad) // stride + 1
        return l


class STL(nn.Module):
    """
    inputs --- [N, token_embedding_size//2]
    """

    def __init__(self, hp, token_embedding_size):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, token_embedding_size // hp.num_heads))
        d_q = token_embedding_size // 2
        d_k = token_embedding_size // hp.num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=token_embedding_size,
            num_heads=hp.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size//num_heads]
        # print(query.shape, keys.shape)
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class GST(nn.Module):
    def __init__(self, hp, token_embedding_size, classes_):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        self.stl = STL(hp, token_embedding_size)

        self.categorical_layer = nn.Linear(token_embedding_size, classes_)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        # print(enc_out.shape)
        style_embed = self.stl(enc_out)

        cat_prob = F.softmax(self.categorical_layer(style_embed.squeeze(0)), dim=-1)
        # print(style_embed.shape, cat_prob.shape)
        return (style_embed.squeeze(0), cat_prob)


class GST_VAE(nn.Module):
    def __init__(self, hp, token_embedding_size, classes_):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        self.stl = STL(hp, token_embedding_size)

        self.mean_linear = nn.Linear(token_embedding_size, hp.vae_size)
        self.logvar_linear = nn.Linear(token_embedding_size, hp.vae_size)
        self.style_embedding = nn.Linear(hp.vae_size, token_embedding_size)
        self.categorical_layer = nn.Linear(token_embedding_size, classes_)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        style_embed = self.stl(enc_out)

        latent_mean = self.mean_linear(style_embed)
        latent_logvar = self.logvar_linear(style_embed)
        std = torch.exp(0.5 * latent_logvar)
        eps = torch.randn_like(std)
        ze = self.style_embedding(eps.mul(std).add_(latent_mean))

        cat_prob = F.softmax(self.categorical_layer(ze), dim=-1)
        # print(ze.unsqueeze(0).shape, cat_prob.shape)
        return (ze, (latent_mean, latent_logvar, cat_prob))


class VAE(nn.Module):
    def __init__(self, hp, token_embedding_size, classes_):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)

        self.mean_linear = nn.Linear(hp.ref_enc_gru_size, hp.vae_size)
        self.logvar_linear = nn.Linear(hp.ref_enc_gru_size, hp.vae_size)
        self.style_embedding = nn.Linear(hp.vae_size, token_embedding_size)
        self.categorical_layer = nn.Linear(token_embedding_size, classes_)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)

        latent_mean = self.mean_linear(enc_out)
        latent_logvar = self.logvar_linear(enc_out)
        std = torch.exp(0.5 * latent_logvar)
        eps = torch.randn_like(std)
        z = self.style_embedding(eps.mul(std).add_(latent_mean))
        cat_prob = F.softmax(self.categorical_layer(z), dim=-1)

        return (z, (latent_mean, latent_logvar, cat_prob))


class GMVAE(nn.Module):
    def __init__(self, hp, token_embedding_size, classes_):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        self.gmvae = GMVAENet(hp.ref_enc_gru_size, token_embedding_size, classes_)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)

        (z, (z, mu, var, y_mu, y_var, prob, logits)) = self.gmvae(enc_out)
        # print(out['prob_cat'].shape, out['logits'].shape)

        return (z, (z, mu, var, y_mu, y_var, prob, logits))


class X_vector(nn.Module):
    def __init__(self, hp, token_embedding_size, num_classes):
        super(X_vector, self).__init__()

        self.input_dim = hp.n_mel_channels
        self.output_dim = token_embedding_size
        self.num_classes = num_classes
        self.layer1 = TDNN_cpu([-2, 2], self.input_dim, self.output_dim, full_context=True)
        self.layer2 = TDNN_cpu([-2, 1, 2], self.output_dim, self.output_dim, full_context=True)
        self.layer3 = TDNN_cpu([-3, 1, 3], self.output_dim, self.output_dim, full_context=True)
        self.layer4 = TDNN_cpu([1], self.output_dim, self.output_dim, full_context=True)
        self.layer5 = TDNN_cpu([1], self.output_dim, 1500, full_context=True)
        self.statpool_layer = StatsPooling()
        self.FF = FullyConnected(self.output_dim)
        self.last_layer = nn.Linear(self.output_dim, self.num_classes)

    def forward(self, x, input_lengths=None):
        x = x.permute(0, 2, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.statpool_layer(x)
        embedding = self.FF(x)
        prob_ = self.last_layer(embedding)

        return embedding, prob_