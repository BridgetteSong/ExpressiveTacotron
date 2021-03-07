import math

import torch
from torch import nn
from torch.nn import functional as F

from modules import SpeakerEncoderNetwork, ExpressiveEncoderNetwork
from text import invalid_phonemes


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, dtype=torch.long).cuda()
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class ZoneOutCell(torch.nn.Module):
    # Adapted from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/tacotron2/decoder.py
    """ZoneOut Cell module.
    This is a module of zoneout described in
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`_.
    This code is modified from `eladhoffer/seq2seq.pytorch`_.
    Examples:
        >> lstm = torch.nn.LSTMCell(16, 32)
        >> lstm = ZoneOutCell(lstm, 0.5)
    .. _`Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`:
        https://arxiv.org/abs/1606.01305
    .. _`eladhoffer/seq2seq.pytorch`:
        https://github.com/eladhoffer/seq2seq.pytorch
    """

    def __init__(self, cell, zoneout_rate=0.1):
        """Initialize zone out cell module.
        Args:
            cell (torch.nn.Module): Pytorch recurrent cell module
                e.g. `torch.nn.Module.LSTMCell`.
            zoneout_rate (float, optional): Probability of zoneout from 0.0 to 1.0.
        """
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_rate = zoneout_rate
        if zoneout_rate > 1.0 or zoneout_rate < 0.0:
            raise ValueError(
                "zoneout probability must be in the range from 0.0 to 1.0."
            )

    def forward(self, inputs, hidden):
        """Calculate forward propagation.
        Args:
            inputs (Tensor): Batch of input tensor (B, input_size).
            hidden (tuple):
                - Tensor: Batch of initial hidden states (B, hidden_size).
                - Tensor: Batch of initial cell states (B, hidden_size).
        Returns:
            tuple:
                - Tensor: Batch of next hidden states (B, hidden_size).
                - Tensor: Batch of next cell states (B, hidden_size).
        """
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_rate)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple(
                [self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)]
            )

        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)
        self.W2.bias.data.fill_(-1.0)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (-g + 1.0) * x
        return y


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        """
        :param attention_rnn_dim: prenet(query) dims
        :param embedding_dim: encoder_seq dims
        :param attention_dim: attention dims
        :param attention_location_n_filters: conv number filters for previous alignmnents
        :param attention_location_kernel_size: conv kernel size
        """
        super(LocationSensitiveAttention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class ContentAttention(nn.Module):
    def __init__(self, query_dim, memory_dim, attention_dim):
        """
        :param query_dim: query dim
        :param memory_dim: key dim
        :param attention_dim: attention dim
        """
        super(ContentAttention, self).__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.attention_dim = attention_dim
        self.query_layer = LinearNorm(query_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(memory_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.score_mask_value = -float("inf")
    
    def forward(self, query, memory, mask=None):
        """
        :param query: decoder query [B, query_dim]
        :param memory: keys [B, seq_len, encoder_hidden_dim]
        :param mask: alignments mask [B, seq_len]
        :return: context [B, encoder_hidden_dim]
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_memory = self.memory_layer(memory)
        energies = self.v(torch.tanh(processed_query + processed_memory))
        alignment = energies.squeeze(2)
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
 
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class ConvPostnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, hparams):
        super(ConvPostnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.post_dropout = hparams.p_postnet_dropout
        in_channels = [hparams.n_mel_channels] + hparams.postnet_embedding_dims
        for i in range(len(in_channels)):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(in_channels[i], hparams.postnet_embedding_dims[i],
                             kernel_size=hparams.postnet_kernel_sizes[i], stride=1,
                             padding=int((hparams.postnet_kernel_sizes[i] - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dims[i]))
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dims[-1], hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_sizes[-1], stride=1,
                         padding=int((hparams.postnet_kernel_sizes[-1] - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), self.post_dropout, self.training)
        x = F.dropout(self.convolutions[-1](x), self.post_dropout, self.training)
        return x


class CBHG(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.postnet_k = hparams.postnet_k
        self.in_channel = hparams.n_mel_channels
        self.postnet_num_highways = hparams.postnet_num_highways
        self.post_projections = hparams.post_projections
        self.bank_kernels = [i for i in range(1, self.postnet_k+1, 2)]
        self.conv_bank = nn.ModuleList()
        for k in self.bank_kernels:
            self.conv_bank += [
                torch.nn.Sequential(
                    ConvNorm(self.in_channel, self.in_channel, k, stride=1, padding=int((k - 1) / 2), dilation=1, w_init_gain='linear'),
                    nn.BatchNorm1d(self.in_channel),
                    nn.ReLU(),
                )
            ]

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.projections = torch.nn.Sequential(
            ConvNorm(len(self.bank_kernels) * self.in_channel, self.post_projections[0], 3, stride=1, padding=1, dilation=1, w_init_gain='linear'),
            nn.BatchNorm1d(self.post_projections[0]),
            nn.ReLU(),
            ConvNorm(self.post_projections[0], self.post_projections[1], 3, stride=1, padding=1, dilation=1, w_init_gain='linear'),
            nn.BatchNorm1d(self.post_projections[1]),
        )

        self.highways = nn.ModuleList()
        for _ in range(self.postnet_num_highways):
            hn = HighwayNetwork(self.post_projections[-1])
            self.highways.append(hn)

        self.rnn = nn.GRU(self.post_projections[-1], self.post_projections[-1], batch_first=True, bidirectional=True)
        self.out = LinearNorm(self.post_projections[-1]*2, self.in_channel)

    def forward(self, x):
        residual = x
        convs = []
        for k in range(len(self.bank_kernels)):
            convs += [self.conv_bank[k](x)]
        convs = torch.cat(convs, dim=1)
        x = self.maxpool(convs)
        x = self.projections(x)
        
        x = x + residual
        x = x.transpose(1, 2)
        for h in self.highways:
            x = h(x)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = self.out(x)

        return x.transpose(1, 2)


class Encoder(nn.Module):
    """Encoder + SkipEncoder:
        - Encoder: no Activation Function https://arxiv.org/abs/2010.04301
        - Repalced with ReLU Activation Function
        - SkipEncoder: https://arxiv.org/pdf/1909.01700.pdf
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.use_skip = hparams.use_skip
        self.embedding_phone = nn.Embedding(hparams.num_chars, hparams.encoder_embedding_dim)
        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim, hparams.encoder_embedding_dim // 2, 1,
                            batch_first=True, bidirectional=True)

       
    def forward(self, phones, input_lengths, actual_phone_lengths):
        """
        :param phones: phones index
        :param input_lengths: input lengths for mask
        :param actual_phone_lengths: actual phone lengths
        :return: encoder_output: skip encoder outputs
        """
        x = self.embedding_phone(phones)
        xs = x
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu().numpy()
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=total_length)

        if not self.use_skip:
            return outputs

        new_outputs = outputs.new_zeros([outputs.size(0), actual_phone_lengths.max().item(), outputs.size(2)])
        for i in range(outputs.size(0)):
            index = torch.where(phones[i] >= len(invalid_phonemes))
            new_outputs[i, :actual_phone_lengths[i].item()] = outputs[i][index]

        return new_outputs

    def inference(self, phones, actual_phone_lengths):
        """
        :param phones: phones index
        :param actual_phone_lengths: actual phone lengths
        :return: encoder_output: skip encoder outputs
        """
        x = self.embedding_phone(phones)
        xs = x
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        if not self.use_skip:
            return outputs

        # Skip Encoder
        new_outputs = outputs.new_zeros([outputs.size(0), actual_phone_lengths.max().item(), outputs.size(2)])
        for i in range(outputs.size(0)):
            index = torch.where(phones[i] >= len(invalid_phonemes))
            new_outputs[i, :actual_phone_lengths[i].item()] = outputs[i][index]

        return new_outputs


class GaussianUpsampling(nn.Module):
    """
    Non-attention Tacotron:
        - https://arxiv.org/abs/2010.04301
    """
    def __init__(self, hparams):
        super(GaussianUpsampling, self).__init__()
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.speaker_embedding_size
        durations_in_channels = [1, hparams.pos_embedding]
        durations_out_channels = [hparams.pos_embedding, hparams.pos_embedding*2]

        self.duration_convs = nn.ModuleList()
        for i in range(len(durations_in_channels)):
            self.duration_convs.append(
                nn.Sequential(
                    ConvNorm(durations_in_channels[i], durations_out_channels[i],
                             kernel_size=hparams.duration_kernel_size,
                             padding=hparams.duration_kernel_size//2),
                    nn.BatchNorm1d(durations_out_channels[i]),
                )
            )

        self.range_parameter_layer = nn.GRU(self.encoder_embedding_dim + durations_out_channels[-1],
                                            (self.encoder_embedding_dim + durations_out_channels[-1])//2,
                                            1, batch_first=True, bidirectional=True)
        self.range_dense = LinearNorm(self.encoder_embedding_dim + durations_out_channels[-1], 1, bias=False, w_init_gain='tanh')
        self.mask_score = -1e15

    def forward(self, encoder_outputs, durations, frames_positions, input_lengths):
        """ Gaussian upsampling
        PARAMS
        ------
        encoder_outputs: Encoder outputs  [B, N, H]
        durations: phoneme durations  [B, N]
        frames_positions: Transformer-styled frames_positions [B, T, pos_embed]
        input_lengths: for text masks
        RETURNS
        -------
        encoder_upsampling_outputs: upsampled encoder_output  [B, T, H]
        """
        B = encoder_outputs.size(0)
        N = encoder_outputs.size(1)
        # total_decoder_steps = torch.max(torch.sum(durations, dim=1)).item()
        total_decoder_steps = frames_positions.size(1)
        c = torch.cumsum(durations, dim=1, dtype=torch.float) - 0.5*durations
        c = c.unsqueeze(2)
        t = torch.arange(total_decoder_steps).expand(B, N, total_decoder_steps).float().cuda()  # [B, N, T]

        # calculate range parameters using ConvNorm and GRU net
        self.range_parameter_layer.flatten_parameters()
        processed_durations = durations.float().unsqueeze(1)
        for duration_conv in self.duration_convs:
            processed_durations = F.dropout(F.gelu(duration_conv(processed_durations)), 0.5, self.training)

        range_parameters, _  = self.range_parameter_layer(
            torch.cat((encoder_outputs, processed_durations.transpose(1, 2)), dim=2))
        range_parameters = F.softplus(self.range_dense(range_parameters))
        
        w_t = -torch.pow((t-c)/range_parameters, 2)
        if input_lengths is not None:
            input_masks = ~get_mask_from_lengths(input_lengths, N) # [B, N]
            masks = input_masks.unsqueeze(2)
            w_t.data.masked_fill_(masks, self.mask_score)
        w_t = F.softmax(w_t, dim=1)
        encoder_upsampling_outputs = torch.bmm(w_t.transpose(1, 2), encoder_outputs)  # [B, T, encoder_hidden_size]
        encoder_upsampling_outputs = torch.cat((encoder_upsampling_outputs, frames_positions), dim=2)

        return encoder_upsampling_outputs

    def inference(self, encoder_outputs, durations, frames_positions):
        """ Gaussian upsampling
        PARAMS
        ------
        encoder_outputs: Encoder outputs  [B, N, H]
        durations: phoneme durations  [B, N]
        frames_positions: Transformer-styled frames_positions [B, T, pos_embed]
        RETURNS
        -------
        encoder_upsampling_outputs: upsampled encoder_output  [B, T, H]
        """
        B = encoder_outputs.size(0)
        N = encoder_outputs.size(1)
        T = torch.sum(durations, dim=1).item()
        c = torch.cumsum(durations, dim=1).float() - 0.5*durations
        c = c.unsqueeze(2)
        t = torch.arange(T).expand(B, N, T).float().cuda()  # [B, N, T]

        # calculate range parameters using ConvNorm and GRU net
        self.range_parameter_layer.flatten_parameters()
        processed_durations = durations.float().unsqueeze(1)
        for duration_conv in self.duration_convs:
            processed_durations = F.dropout(F.gelu(duration_conv(processed_durations)), 0.5, self.training)

        range_parameters, _  = self.range_parameter_layer(
            torch.cat((encoder_outputs, processed_durations.transpose(1, 2)), dim=2))
        range_parameters = F.softplus(self.range_dense(range_parameters))

        w_t = -torch.pow((t-c)/range_parameters, 2)
        w_t = F.softmax(w_t, dim=1)
        encoder_upsampling_outputs = torch.bmm(w_t.transpose(1, 2), encoder_outputs)  # [B, T, encoder_hidden_size]
        encoder_upsampling_outputs = torch.cat((encoder_upsampling_outputs, frames_positions), dim=2)
        return encoder_upsampling_outputs


class Durian(nn.Module):
    """ Durian
    https://arxiv.org/pdf/1909.01700.pdf
    """
    def __init__(self):
        super(Durian, self).__init__()

    def forward(self, encoder_outputs, durations, frames_positions, input_lengths):
        B, L, D = encoder_outputs.size()
        T = frames_positions.size(1)
        # T = int(torch.sum(durations, dim=-1).max().item())
        expanded = encoder_outputs.new_zeros(B, T, D)
        for i, d in enumerate(durations):
            mel_len = torch.sum(d).item()
            expanded[i, :mel_len] = torch.repeat_interleave(encoder_outputs[i], d, dim=0)

        expanded = torch.cat((expanded, frames_positions), dim=2)
        return expanded

    def inference(self, encoder_outputs, durations, frames_positions):
        B, L, D = encoder_outputs.size()
        T = int(torch.sum(durations, dim=-1).max().item())
        expanded = encoder_outputs.new_zeros(B, T, D)
        for i, d in enumerate(durations):
            mel_len = torch.sum(d).item()
            expanded[i, :mel_len] = torch.repeat_interleave(encoder_outputs[i], d, dim=0)

        expanded = torch.cat((expanded, frames_positions), dim=2)
        return expanded


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.token_embedding_size + hparams.pos_embedding + hparams.speaker_embedding_size
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.attention_dim = hparams.attention_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dims = hparams.prenet_dims
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(self.n_mel_channels * self.n_frames_per_step, self.prenet_dims)

        self.attention_rnn = nn.LSTMCell(self.prenet_dims[-1] + self.encoder_embedding_dim, self.attention_rnn_dim)
        self.decoder_rnn = nn.LSTMCell(self.decoder_rnn_dim + self.encoder_embedding_dim, self.decoder_rnn_dim)

        if self.p_attention_dropout > 0:
            self.attention_rnn = ZoneOutCell(self.attention_rnn, self.p_attention_dropout)
        if self.p_decoder_dropout > 0:
            self.decoder_rnn = ZoneOutCell(self.decoder_rnn, self.p_decoder_dropout)

        self.linear_projection = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            self.n_mel_channels * self.n_frames_per_step)

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = memory.new_zeros(
            B, self.n_mel_channels * self.n_frames_per_step)
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)

        self.attention_hidden = memory.new_zeros(B, self.attention_rnn_dim)
        self.attention_cell = memory.new_zeros(B, self.attention_rnn_dim)

        self.decoder_hidden = memory.new_zeros(B, self.decoder_rnn_dim)
        self.decoder_cell = memory.new_zeros(B, self.decoder_rnn_dim)

        self.attention_context = memory.new_zeros(B, self.encoder_embedding_dim)

        self.memory = memory
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.contiguous().view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:

        RETURNS
        -------
        mel_outputs:
        """
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs

    def decode(self, decoder_input, index):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        index: current mel index

        RETURNS
        -------
        mel_output:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden,self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_context = self.memory[:, index:index+self.n_frames_per_step, :].mean(dim=1)

        decoder_input = torch.cat((self.attention_hidden, self.attention_context), 1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        return decoder_output

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs [B, n_mels, T_out]
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths, max_len=memory.size(1)))

        mel_outputs = []

        for i in range(0, decoder_inputs.size(0) - 1, 1):
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output = self.decode(decoder_input, self.n_frames_per_step*i)
            mel_outputs += [mel_output]

        mel_outputs = self.parse_decoder_outputs(mel_outputs)

        return mel_outputs

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs = []
        for index in range(0, memory.size(1), self.n_frames_per_step):
            decoder_input = self.prenet(decoder_input)
            mel_output = self.decode(decoder_input, index)
            mel_outputs += [mel_output]
            decoder_input = mel_output

        mel_outputs = self.parse_decoder_outputs(mel_outputs)

        return mel_outputs


class Tacotron2(nn.Module):
    """
    Non Attention Expressive-Tacotron2 with the CBHG postnet
    """
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.hparams = hparams
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.adaption = hparams.adaption
        self.encoder = Encoder(hparams)
        self.align = GaussianUpsampling(hparams)
        # self.align = Durian()
        self.decoder = Decoder(hparams)
        self.postnet = CBHG(hparams)

        self.spk_embedding = nn.Embedding(hparams.speaker_classes, hparams.speaker_embedding_size)
        self.expressive_encoder = ExpressiveEncoderNetwork(hparams)

        if self.adaption:
            for i in self.encoder.parameters():
                i.requires_grad = False

    def parse_output(self, mel_outputs, mel_outputs_postnet, output_lengths=None, max_output_length=None):
        if output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, max_output_length)
            mask = mask.unsqueeze(1)

            mel_outputs.data.masked_fill_(mask, 0.0)
            mel_outputs_postnet.data.masked_fill_(mask, 0.0)

        return mel_outputs, mel_outputs_postnet

    def forward(self, phones, mels, x_spk, durations, frames_positions, input_lengths, actual_phone_lengths, mel_lengths):
        encoder_outputs = self.encoder(phones, input_lengths, actual_phone_lengths)
        spk_embed = self.spk_embedding(x_spk.cuda())
        spk_embed = spk_embed.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
        encoder_outputs = torch.cat((encoder_outputs, spk_embed), 2)

        # for Gaussian Upsampling
        encoder_expand = self.align(encoder_outputs, durations, frames_positions, input_lengths)

        # GMVAE
        (expressive_embedding, e_prob) = self.expressive_encoder(mels.transpose(1, 2), input_lengths=None)
        expressive_embedding = expressive_embedding.unsqueeze(1).expand(
            encoder_expand.size(0), encoder_expand.size(1), -1)
        encoder_expand = torch.cat((encoder_expand, expressive_embedding), 2)

        mel_outputs = self.decoder(encoder_expand, mels, input_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # mask mel is not convergenced
        # mel_outputs, mel_outputs_postnet = self.parse_output(
        #     mel_outputs, mel_outputs_postnet, mel_lengths, mel_outputs.size(-1))

        return [mel_outputs, mel_outputs_postnet, e_prob]

    def inference(self, phones, actual_phone_len, x_spk, mels, durations):
        phones = torch.tensor(phones, dtype=torch.long).unsqueeze(0).cuda()
        x_spk = torch.tensor(x_spk, dtype=torch.long).unsqueeze(0).cuda()
        actual_phone_len = torch.tensor(actual_phone_len, dtype=torch.long).unsqueeze(0).cuda()
        mels = torch.tensor(mels).cuda().unsqueeze(0)

        encoder_outputs = self.encoder.inference(phones, actual_phone_len)
        spk_embed = self.spk_embedding(x_spk)
        spk_embed = spk_embed.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
        encoder_outputs = torch.cat((encoder_outputs, spk_embed), 2)

        # durations and frame positions
        tmp_pes = []
        print(durations)
        if durations.sum() % self.n_frames_per_step != 0:
            durations[-1] += self.n_frames_per_step - durations.sum() % self.n_frames_per_step

        for i, d in enumerate(durations, 1):
            pe = torch.zeros([d, self.hparams.pos_embedding]).cuda()
            position = torch.arange(0, d).view(d, 1).cuda()
            div_term = torch.exp(torch.arange(0, self.hparams.pos_embedding, 2) * -(
                        math.log(10000.0) / self.hparams.pos_embedding)).cuda()
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            tmp_pes.append(pe)
        pes = torch.cat(tmp_pes, 0).unsqueeze(0)
        durations = durations.unsqueeze(0)

        # for Gaussian Upsampling
        encoder_expand = self.align.inference(encoder_outputs, durations, pes)

        # expressive_encoder
        (expressive_embedding, e_prob) = self.expressive_encoder(mels, input_lengths=None)
        expressive_embedding = expressive_embedding.unsqueeze(1).expand(
            encoder_expand.size(0), encoder_expand.size(1), -1)
        encoder_expand = torch.cat((encoder_expand, expressive_embedding), 2)

        mel_outputs = self.decoder.inference(encoder_expand)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs.squeeze(0), mel_outputs_postnet.squeeze(0)
