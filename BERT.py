import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils.vocab import *
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding


class Embedding(nn.Module):
    def __init__(self, dim, num_vocab, max_len):
        super().__init__()

        self.token_embed = nn.Embedding(num_vocab, dim)
        self.pos_embed = FreqEmbedding(max_len, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        embed = self.token_embed(x)

        pos = self.pos_embed(x)
        pos = pos.type_as(embed)

        embed = self.norm(embed + pos)

        return embed


class FreqEmbedding(nn.Module):
    """
    Refer to https://github.com/dreamgonfly/transformer-pytorch/blob/master/embeddings.py
    """

    def __init__(self, max_len, dim):
        super().__init__()

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, dim)
        pe.require_grad = False

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = (torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, : x.shape[1]]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):
    """
    Refer to https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    """

    def __init__(self, dim, heads=12, dim_head=64):
        super().__init__()

        self.heads = heads
        self.scale = dim_head**-0.5
        hidden_dim = dim_head * heads

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(dim=32)

    def forward(self, x, mask=None):
        # x -> (batch (b), seq (n), dim (d))
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # rotary embedding
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = mask.unsqueeze(1)

        if mask is not None:
            fill_value = 1e-9 if dots.dtype == torch.float32 else 1e-4
            dots.masked_fill_(mask, fill_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    """
    Refer to https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, rate):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([])
        self.dropout = nn.Dropout(rate)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x, mask=None):
        h_list = []

        for attn, ff in self.layers:
            x = self.norm1(self.dropout(attn(x, mask)) + x)
            x = self.norm2(self.dropout(ff(x)) + x)
            h_list.append(x[:, 0])

        return x, h_list


class BERT(nn.Module):
    def __init__(self, dim, vocab, depth, heads, dim_head, mlp_dim, max_len, rate):
        super().__init__()

        self.vocab = vocab
        num_vocab = len(vocab)

        self.embedding = Embedding(dim, num_vocab, max_len)

        self.transformer = Transformer(
            dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            rate=rate,
        )

        self.linear_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_vocab),
        )

        # test whether it is valid
        # decoder is shared with embedding layer
        # self.decoder = nn.Linear(dim, num_vocab, bias=False)
        # self.decoder.weight = self.embedding.token_embed.weight
        # self.decoder_bias = nn.Parameter(torch.zeros(num_vocab))

    def get_attn_pad_mask(self, seq):
        batch_size, len_seq = seq.shape

        pad_idx = self.vocab.to_i(PAD_TOKEN)
        pad_attn_mask = seq.eq(pad_idx).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, len_seq, len_seq)

        return pad_attn_mask

    def forward(self, x):
        attn_mask = self.get_attn_pad_mask(x)

        # Transformers
        x = self.embedding(x)
        h, h_list = self.transformer(x, attn_mask)

        # last layer
        logits = self.linear_head(h)
        # logits = self.decoder(logits) + self.decoder_bias

        return logits, h[:, 0], h_list


class BERT_Lightning(pl.LightningModule):
    # BERT base: L=12, H=768, A=12
    def __init__(
        self,
        dim,
        depth=12,
        heads=12,
        dim_head=64,
        mlp_dim=2048,
        max_len=512,
        rate=0.1,
        loss_weights=[1, 1],
        lr=1e-3,
        warm_up=5000,
        temp=1,
        mode="BERT",
    ):
        super().__init__()

        self.vocab = RemiVocab()

        self.model = BERT(
            dim=dim,
            vocab=self.vocab,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            max_len=max_len,
            rate=rate,
        )

        self.lr = lr
        self.warm_up = warm_up
        self.temp = temp
        self.mode = mode

        self.loss_weights = loss_weights
        self.mask_ce_loss = nn.CrossEntropyLoss(ignore_index=RemiVocab().to_i(PAD_TOKEN))
        self.nce_ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        logits, h, h_list = self.model(x)

        return logits, h, h_list

    def configure_optimizers(self):
        """
        Refer to https://gist.github.com/gautierdag/925760d4295080c1860259dba43e4c01
        """

        opt = AdamW(self.parameters(), lr=self.lr)

        def warm_decay(step):
            if step < self.warm_up:
                return step / self.warm_up
            return self.warm_up**0.5 * step**-0.5

        sch = {
            "scheduler": LambdaLR(opt, warm_decay),
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
        }

        return [opt], [sch]

    def get_acc(self, y_pred, y_true):
        y_pred = nn.Softmax(dim=-1)(y_pred)
        y_pred = y_pred.argmax(-1)

        nonzero_idx = y_true != self.vocab.to_i(PAD_TOKEN)

        nom = (y_pred[nonzero_idx] == y_true[nonzero_idx]).sum()
        denom = y_pred[nonzero_idx].numel()

        return nom / denom

    def compute_nce_loss(self, feat1, feat2):
        """
        Refer to https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """

        labels = torch.cat([torch.arange(feat1.shape[0]) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.type_as(feat1)

        feat = torch.cat([feat1, feat2], dim=0)
        feat = F.normalize(feat, dim=1)
        sim = torch.matmul(feat, feat.T)

        batch_size = feat.shape[0]

        mask = torch.eye(batch_size).bool()
        labels = labels[~mask].view(batch_size, -1).bool()

        sim = sim[~mask].view(batch_size, -1)
        pos = sim[labels].view(batch_size, -1)
        neg = sim[~labels].view(batch_size, -1)

        logits = torch.cat([pos, neg], dim=1) / self.temp
        labels = torch.zeros(logits.shape[0]).type_as(logits).to(torch.long)

        loss = self.nce_ce_loss(logits, labels)

        # get nce accuracy
        pred = torch.argmax(logits, dim=-1)
        acc = (pred == labels).sum() / pred.numel()

        return loss, acc

    def mode_change(self, batch):
        if self.mode == "BERT-aug":
            return batch["x_aug"]

        elif self.mode == "BERT-neighbor":
            return batch["x_neigh"]

        elif self.mode == "BERT-dropout":
            return batch["x_mask"]

    def training_step(self, train_batch, batch_idx):
        y_mask_pred, h_mask_pred, _ = self.model(train_batch["x_mask"])

        if self.mode != "BERT":
            x_pair = self.mode_change(train_batch)
            y_pair_pred, h_pair_pred, _ = self.model(x_pair)

        # MLM loss
        mlm_loss = self.mask_ce_loss(y_mask_pred.transpose(1, 2), train_batch["y_mask"])
        mlm_acc = self.get_acc(y_mask_pred, train_batch["y_mask"])

        nce_loss = 0
        nce_acc = 0

        # contrastive loss
        if self.mode != "BERT":
            nce_loss, nce_acc = self.compute_nce_loss(h_mask_pred, h_pair_pred)

        # total loss
        loss = (self.loss_weights[0] * mlm_loss) + (self.loss_weights[1] * nce_loss)

        batch_size = h_mask_pred.shape[0]
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("train_mlm_loss", mlm_loss, prog_bar=True, batch_size=batch_size)
        self.log("train_mlm_acc", mlm_acc, prog_bar=True, batch_size=batch_size)
        self.log("train_nce_loss", nce_loss, prog_bar=True, batch_size=batch_size)
        self.log("train_nce_acc", nce_acc, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, val_batch, batch_idx):
        y_mask_pred, h_mask_pred, _ = self.model(val_batch["x_mask"])

        if self.mode != "BERT":
            x_pair = self.mode_change(val_batch)
            y_pair_pred, h_pair_pred, _ = self.model(x_pair)

        # MLM loss
        mlm_loss = self.mask_ce_loss(y_mask_pred.transpose(1, 2), val_batch["y_mask"])
        mlm_acc = self.get_acc(y_mask_pred, val_batch["y_mask"])

        nce_loss = 0
        nce_acc = 0

        # contrastive loss
        if self.mode != "BERT":
            nce_loss, nce_acc = self.compute_nce_loss(h_mask_pred, h_pair_pred)

        # total loss
        loss = (self.loss_weights[0] * mlm_loss) + (self.loss_weights[1] * nce_loss)

        batch_size = h_mask_pred.shape[0]
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val_mlm_loss", mlm_loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val_mlm_acc", mlm_acc, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val_nce_loss", nce_loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val_nce_acc", nce_acc, prog_bar=True, batch_size=batch_size, sync_dist=True)

        return loss
