''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer




def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    sz_b, len_s, _ = seq.size()
    return _get_subsequent_mask(len_s, seq.device)


def _get_subsequent_mask(len_s, device):
    ''' For masking out the subsequent info. '''
    subsequent_mask = 1 - torch.triu(
        torch.ones((1, len_s, len_s), device=device, dtype=torch.uint8), diagonal=1)
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, d_word_vec=77, n_layers=2, n_head=1, d_k=16, d_v=16,
            d_model=77, d_inner=16, dropout=0.1, n_position=200, kernel = 'linear', kernel_size_tcn=3, kernel_size_scn = 2):

        super().__init__()

#        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, kernel = kernel, kernel_size_tcn=kernel_size_tcn, kernel_size_scn = kernel_size_scn)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
       
#        add
        enc_output = self.dropout(self.position_enc( src_seq ))
#        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=200, dropout=0.1):

        super().__init__()

#        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
#        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, enc_output, trg_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
#        dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq)))
        dec_output = self.dropout(self.position_enc(trg_seq))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, d_word_vec=77, d_model=77, d_inner=32,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200):

        super().__init__()
        self.d_word_vec = d_word_vec
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.encoder = Encoder(n_position=n_position, d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.decoder = Decoder(n_position=n_position,d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq, trg_seq):
        sz_b, len_s, _ = trg_seq.size()
        trg_mask = get_pad_mask(torch.ones(sz_b, len_s).to(trg_seq.device), 0).int() & get_subsequent_mask(trg_seq).int()
#        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx).int() & get_subsequent_mask(trg_seq).int()
        enc_output, *_ = self.encoder(src_seq)
        dec_output, *_ = self.decoder(trg_seq, enc_output, trg_mask)
        return dec_output
    
    def description(self):
        return 'An entire Transformer model with attention mechanism. \n ' \
               '(parameters: d_word_vec=%s, d_model=%s, d_inner=%s, n_layers=%s, n_head=%s, d_k=%s, d_v=%s)'%\
               (self.d_word_vec, self.d_model, self.d_inner, self.n_layers, self.n_head, self.d_k, self.d_v)
