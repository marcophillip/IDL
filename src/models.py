import torch
from torch import nn
from torch import Tensor
from torch.nn import Transformer
import math
from typing import Iterable, List
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 max_len: int = 2000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000)/ emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, token_embedding: Tensor):
        x = self.pos_embedding[:token_embedding.size(0), :]
        x = token_embedding + x
        
        x = self.dropout(x)
        
        return x
    
    class TokenEmbedding(nn.Module):
        def __init__(self, vocab_size: int, emb_size):
            super(TokenEmbedding, self).__init__()
            self.embedding = nn.Embedding(vocab_size, emb_size)
            self.emb_size = emb_size

        def forward(self, tokens: Tensor):
            return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

    class Seq2Seq(nn.Module):
        def __init__(self,
                     num_encoder_layers: int,
                     num_decoder_layers: int,
                     emb_size: int,
                     nhead: int,
                     src_vocab_size: int,
                     tgt_vocab_size: int,
                     dim_feedforward: int = 512,
                     dropout: float = 0.1):

            super(Seq2Seq, self).__init__()
            self.transformer = Transformer(
                d_model=emb_size,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )

            self.generator = nn.Linear(emb_size, tgt_vocab_size)
            self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
            self.positional_encoding = PositionalEncoding(
                emb_size, dropout=dropout
            )

        def forward(self,
                    src: Tensor,
                    trg: Tensor,
                    src_mask: Tensor,
                    tgt_mask: Tensor,
                    src_padding_mask: Tensor,
                    tgt_padding_mask:Tensor,
                    memory_key_padding_mask: Tensor):

            src_emb = self.positional_encoding(self.src_tok_emb(src))
            trg_emb = self.positional_encoding(self.tgt_tok_emb(trg))
            outs = self.transformer(src_emb,
                                    trg_emb,
                                    tgt_mask,
                                    None,
                                    src_padding_mask,
                                    tgt_padding_mask,
                                    memory_key_padding_mask)
            return self.generator(outs)

        def encode(self, src: Tensor, src_mask: Tensor):
            return self.transformer.encoder(self.positional_encoding(src,
                                                                     self.src_tok_emb(src)), src_mask
                                            )

        def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
            return self.transformer.decoder(self.positional_encoding(
                self.tgt_tok_emb(tgt)), memory, tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2Seq(NUM_ENCODER_LAYERS,
                      NUM_DECODER_LAYERS,
                      EMB_SIZE,
                      NHEAD,
                      SRC_VOCAB_SIZE,
                      TGT_VOCAB_SIZE,
                      FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

from torch.nn.utils.rnn import pad_sequence


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

