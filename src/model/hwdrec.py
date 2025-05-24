import copy
import torch
import torch.nn as nn
import torch.nn.functional  as F
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention

class HWDRecModel(SequentialRecModel):
    def __init__(self, args):
        super(HWDRecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = RecEncoder(args)
        self.apply(self.init_weights)

        self.muilt_level_dwt = MultiLevelHaarWaveletTransform1D(args.hidden_size, levels=3)
        self.highs = []
        self.low_layer = HighFreqDWT1D(args.hidden_size, wavelet='haar', filter_type='low')
        self.fre_LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.fre_dropout = nn.Dropout(args.hidden_dropout_prob)
        # method 1
        self.wave_attention = WaveletAttention(args.hidden_size)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb_w_position = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb_w_position)
        sequence_emb = self.dropout(sequence_emb)

        sequence_emb_for_hf = self.fre_LayerNorm(item_embeddings)
        sequence_emb_for_hf = self.fre_dropout(sequence_emb_for_hf)
        return sequence_emb, sequence_emb_for_hf

    def mask_deal(self, padding_mask):
        padding_mask = (padding_mask > 0).long()
        half_mask = F.max_pool1d(
            padding_mask.unsqueeze(1).float(),  # float type
            kernel_size=2,
            stride=2
        ).squeeze(1).long()  # (B, L//2)
        extended_attention_mask = half_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        max_len = half_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(padding_mask.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return half_mask, extended_attention_mask

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        half_mask, extended_attention_mask = self.mask_deal(input_ids)
        sequence_emb, sequence_emb_wo_pos = self.add_position_embedding(input_ids)
        # high_frequency
        self.highs = self.muilt_level_dwt(half_mask, sequence_emb_wo_pos)
        # low_frequency：1st
        sequence_emb = self.low_layer(sequence_emb)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]
        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        # Transformer output
        seq_output = self.forward(input_ids)
        # HF output
        high_features = torch.cat(self.highs, dim=1)
        high_attention_output = self.wave_attention(seq_output, high_features)
        high_attention_output = high_attention_output[:, -1, :]
        seq_output = seq_output[:, -1, :]

        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        high_attention_output = high_attention_output + high_features.mean(dim=1)
        dsp_logits = torch.matmul(high_attention_output, item_emb.transpose(0, 1))
        dsp_loss = nn.CrossEntropyLoss()(dsp_logits, answers)
        return loss * (1 - self.args.dsp_loss_weight) + dsp_loss * self.args.dsp_loss_weight


class WaveletAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

    def forward(self, q, wavelet_features):
        # q: (batch, 1, dim)
        # wavelet_features: (batch, 3, dim)

        # 1. projection
        q_trans = self.q_proj(q)  # (batch, 1, dim)
        k = self.k_proj(wavelet_features)  # (batch, 3, dim)

        # 2. attention score
        attn_scores = torch.matmul(q_trans, k.transpose(1, 2))  # (batch, 1, 3)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, 3)

        # 3. merge
        weighted_wavelet = torch.matmul(attn_weights, wavelet_features)  # (batch, 1, dim)

        return weighted_wavelet


class RecEncoder(nn.Module):
    def __init__(self, args):
        super(RecEncoder, self).__init__()
        self.args = args
        block = RecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class RecBlock(nn.Module):
    def __init__(self, args):
        super(RecBlock, self).__init__()
        self.layer = RecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class RecLayer(nn.Module):
    def __init__(self, args):
        super(RecLayer, self).__init__()
        self.args = args
        self.attention_layer = MultiHeadAttention(args)

    def forward(self, input_tensor, attention_mask):
        gsp = self.attention_layer(input_tensor, attention_mask)
        return gsp


class HighFreqDWT1D(nn.Module):
    def __init__(self, dim, wavelet='haar', filter_type='low'):
        super().__init__()
        self.dim = dim

        if wavelet == 'haar':
            if filter_type == 'high':
                h_filter = torch.tensor([-1.0, 1.0], requires_grad=False) / torch.sqrt(torch.tensor(2.0))  # 高通滤波器
            else:
                h_filter = torch.tensor([1.0, 1.0], requires_grad=False) / torch.sqrt(torch.tensor(2.0))
        else:
            raise NotImplementedError("No Supported Wavelet")

        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=2,
            stride=2,
            groups=dim,
            bias=False
        )

        with torch.no_grad():
            self.conv.weight.data = h_filter.view(1, 1, -1).repeat(dim, 1, 1)  # 形状 (dim, 1, 2)

    def forward(self, x):
        """
        input: (batch, seq_len, dim)
        output: (batch, seq_len//2, dim)
        """
        # -> (batch, dim, seq_len)
        x = x.permute(0, 2, 1)

        # -> (batch, dim, seq_len//2)
        high = self.conv(x)

        # -> (batch, seq_len//2, dim)
        return high.permute(0, 2, 1)


class MultiLevelHaarWaveletTransform1D(nn.Module):
    def __init__(self, dim, levels):
        super(MultiLevelHaarWaveletTransform1D, self).__init__()
        self.levels = levels
        # Haar Wave
        low_pass_filter = torch.tensor([-1.0, 1.0], requires_grad=False) / torch.sqrt(torch.tensor(2.0))
        high_pass_filter = torch.tensor([1.0, 1.0], requires_grad=False) / torch.sqrt(torch.tensor(2.0))

        self.low_pass_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=2, stride=2, groups=dim,
                                       bias=False)
        self.high_pass_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=2, stride=2, groups=dim,
                                        bias=False)

        with torch.no_grad():
            self.low_pass_conv.weight.data = low_pass_filter.view(1, 1, -1).repeat(dim, 1, 1)  # 形状 (dim, 1, 2)
            self.high_pass_conv.weight.data = high_pass_filter.view(1, 1, -1).repeat(dim, 1, 1)  # 形状 (dim, 1, 2)

    def forward(self, half_mask, x):
        # (batch, seq_len, dim) -> (batch, dim, seq_len)
        x = x.permute(0, 2, 1)
        coeffs = []
        current_x = x
        for level in range(self.levels):
            low_pass = self.low_pass_conv(current_x)  # shape: (batch, dim, seq_len // 2^(level+1))

            high_pass = self.high_pass_conv(current_x)  # shape: (batch, dim, seq_len // 2^(level+1))
            high_pass = high_pass.permute(0, 2, 1)  # (batch, seq_len // 2^(level+1), dim)
            high_pass = high_pass * half_mask.unsqueeze(-1)
            high_pass = high_pass.sum(dim=1, keepdim=True) / (half_mask.sum(dim=1, keepdim=True) + 1e-12).unsqueeze(-1)

            coeffs.append(high_pass)

            current_x = low_pass

            half_mask = F.max_pool1d(
                half_mask.unsqueeze(1).float(),
                kernel_size=2,
                stride=2
            ).squeeze(1).long()

        return coeffs