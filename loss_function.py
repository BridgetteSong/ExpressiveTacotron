# from https://github.com/NVIDIA/tacotron2
# Modified by Ajinkya Kulkarni
import math

from torch import nn
import numpy as np
import torch
from torch.nn import functional as F

eps = 1e-8


class Tacotron2Loss(nn.Module):
    def __init__(self, hp, update_step):
        super(Tacotron2Loss, self).__init__()
        self.expressive_classes = hp.emotion_classes
        self.speaker_classes = hp.speaker_classes
        self.cat_lambda = hp.cat_lambda
        self.speaker_encoder_type = hp.speaker_encoder_type
        self.expressive_encoder_type = hp.expressive_encoder_type
        self.model_type = hp.model_type
        self.update_step = update_step
        self.kl_lambda = hp.kl_lambda
        self.kl_incr = hp.kl_incr
        self.kl_step = hp.kl_step
        self.kl_step_after = hp.kl_step_after
        self.kl_max_step = hp.kl_max_step

        self.cat_incr = hp.cat_incr
        self.cat_step = hp.cat_step
        self.cat_step_after = hp.cat_step_after
        self.cat_max_step = hp.cat_max_step

    def get_w(self, T, N):
        g = 0.2
        w = torch.zeros((T, N)).cuda()
        for t in range(T):
            for n in range(N):
                w[t, n] = 1 - math.exp(-(n / N - t / T) * (n / N - t / T) / (2 * g * g))
        return w

    def guided_attention_loss(self, attention):
        w = self.get_w(attention.size(1), attention.size(2))
        loss = torch.mean(w * attention)
        return loss

    def indices_to_one_hot(self, data, n_classes):
        targets = np.array(data).reshape(-1)
        return torch.from_numpy(np.eye(n_classes)[targets]).cuda()
        # targets = data.contiguous().view(-1)
        # return torch.eye(targets, device=targets.device)[n_classes]


    def KL_loss(self, mu, var):
        return torch.mean(0.5 * torch.sum(torch.exp(var) + mu ** 2 - 1. - var, 1))

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        # KL loss is right?
        """
        KL_Loss = sum(p(x))*(log(q(x))-log(p(x)))
        """
        # loss = (self.log_normal2(z, z_mu, z_var) - self.log_normal2(z, z_mu_prior, z_var_prior))
        # loss *= torch.sum(torch.exp(-0.5*torch.pow(z - z_mu_prior, 2) / var)/torch.sqrt(2.0 * np.pi * z_var_prior), dim=-1)
        return loss.mean()

    def get_encoder_loss(self, id_, prob_, classes_, cat_lambda, kl_lambda, encoder_type):
        cat_target = self.indices_to_one_hot(id_, classes_)

        if (encoder_type == 'gst' or encoder_type == 'x-vector') and cat_lambda != 0.0:
            loss = cat_lambda * (-self.entropy(cat_target, prob_) - np.log(0.1))
        elif (encoder_type == 'vae' or encoder_type == 'gst_vae') and (cat_lambda != 0.0 or kl_lambda != 0.0):
            loss = cat_lambda * (-self.entropy(cat_target, prob_[2]) - np.log(0.1)) + \
                   kl_lambda * self.KL_loss(prob_[0], prob_[1])
        elif encoder_type == 'gmvae' and (cat_lambda != 0.0 or kl_lambda != 0.0):
            loss = self.gaussian_loss(prob_[0], prob_[1], prob_[2], prob_[3], prob_[4])*kl_lambda + (-self.entropy(cat_target, prob_[5]) - np.log(0.1))*cat_lambda
        else:
            loss = 0.0

        return loss

    def update_lambda(self, iteration):
        iteration += 1
        if self.update_step % iteration == 0:
            self.kl_lambda = self.kl_lambda + self.kl_incr
            self.cat_lambda = self.cat_lambda + self.cat_incr

        if iteration <= self.kl_max_step and iteration % self.kl_step == 0:
            kl_lambda = self.kl_lambda
        elif iteration > self.kl_max_step and iteration % self.kl_step_after == 0:
            kl_lambda = self.kl_lambda
        else:
            kl_lambda = 0.0

        if iteration <= self.cat_max_step and iteration % self.cat_step == 0:
            cat_lambda = self.cat_lambda
        elif iteration > self.cat_max_step and iteration % self.cat_step_after == 0:
            cat_lambda = self.cat_lambda
        else:
            cat_lambda = 0.0

        return min(1, kl_lambda), min(1, cat_lambda)

    def log_normal(self, x, mu, var):
        if eps > 0.0:
            var = var + eps
        return -0.5 * torch.sum(np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def log_normal2(self, x, mu, var):
        if eps > 0.0:
            var = var + eps
        return -0.5 * (np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)

    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))

    def forward(self, iteration, model_output, targets, e_id):

        kl_lambda, cat_lambda = self.update_lambda(iteration)

        # tacotron losses
        if self.model_type == "attention":
            mel_target, gate_target = targets[0], targets[1]
            gate_target = gate_target.view(-1, 1)
            mel_outputs, mel_outputs_postnet, gate_out, alignments, e_prob = model_output
            gate_out = gate_out.view(-1, 1)
            gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
            align_loss = self.guided_attention_loss(alignments)
        # non-attention tacotron losses
        elif self.model_type == "non_attention":
            mel_target = targets
            mel_outputs, mel_outputs_postnet, e_prob = model_output
            gate_loss, align_loss = 0.0, 0.0
        else:
            raise ValueError("unsupported model type")

        # public lossed
        l1_criterion = F.l1_loss(mel_outputs, mel_target) + F.l1_loss(mel_outputs_postnet, mel_target)
        mel_criterion = F.mse_loss(mel_outputs, mel_target) + F.mse_loss(mel_outputs_postnet, mel_target)

        # speaker_encoder_loss
        speaker_loss = 0.0
        # speaker_loss = self.get_encoder_loss(s_id, s_prob, self.speaker_classes, cat_lambda, kl_lambda,
        #                                      self.speaker_encoder_type) if self.emotioned else 0.0

        # expressive_encoder_loss
        expressive_loss = self.get_encoder_loss(e_id, e_prob, self.expressive_classes, cat_lambda, kl_lambda,
                                                self.expressive_encoder_type)

        return l1_criterion + mel_criterion + gate_loss + align_loss + speaker_loss + expressive_loss