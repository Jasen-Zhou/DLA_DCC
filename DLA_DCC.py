import copy
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)


class DynamicLinBertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, seq_len, r_min=16, r_max=64, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.seq_len = seq_len
        self.r_min = r_min
        self.r_max = r_max

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.proj_k = nn.Parameter(torch.randn(num_heads, seq_len, r_max))
        self.proj_v = nn.Parameter(torch.randn(num_heads, seq_len, r_max))
        nn.init.xavier_uniform_(self.proj_k)
        nn.init.xavier_uniform_(self.proj_v)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=True,
    ):
        bs, seq_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self.transpose_for_scores(query)  # [bs, num_heads, seq_len, head_dim]
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        batch_var = hidden_states.var(dim=[1, 2])  # [bs]
        var_norm = (batch_var - batch_var.min()) / (batch_var.max() - batch_var.min() + 1e-6)
        curr_r = int((self.r_min + var_norm.mean() * (self.r_max - self.r_min)).round().item())

        curr_proj_k = self.proj_k[..., :curr_r]  # [num_heads, seq_len, curr_r]
        curr_proj_v = self.proj_v[..., :curr_r]

        key_proj = torch.einsum('bhld,hlk->bhkd', key, curr_proj_k)   # (bs, num_heads, curr_r, head_dim)
        value_proj = torch.einsum('bhld,hlk->bhkd', value, curr_proj_v)

        attn_scores = torch.matmul(query, key_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [bs, num_heads, seq_len, curr_r]

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, value_proj)  # [bs, num_heads, seq_len, head_dim]
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(bs, seq_len, self.hidden_size)

        output = self.out_proj(context)
        output = self.out_dropout(output)

        return (output, attn_probs if output_attentions else None, curr_r)


model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=2)
for i in range(12):
    model.bert.encoder.layer[i].attention.self = DynamicLinBertSelfAttention(
        hidden_size=768, num_heads=12, seq_len=128, r_min=16, r_max=64, dropout=0.1
    )


tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
dataset   = load_dataset('stanfordnlp/imdb')

def map_to_raw(examples):
    return {"raw_text": examples["text"], "labels": examples["label"]}

raw_ds = dataset.map(
    map_to_raw,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Building raw_text dataset"
)
raw_ds.set_format(type="python")

train_dataset = raw_ds["train"]
eval_dataset  = raw_ds["test"]

# 断言：样本里必须有 raw_text
sample = train_dataset[0]
assert "raw_text" in sample and isinstance(sample["raw_text"], str), f"raw_text missing, keys={list(sample.keys())}"

# =========================
# 文本污染函数
# =========================
QWERTY = {"a":"qwsz","e":"wsrd34","i":"uok89","o":"ipk90","t":"rfgy5"}
HOMO   = {"a":"а","e":"е","o":"о","c":"с"}  # 拉丁->西里尔形近字
OCR_MAP= {"rn":"m","m":"rn","0":"O","1":"l"}

def keyboard_typo(s, rate=0.02):
    out=[]
    for ch in s:
        out.append(random.choice(QWERTY[ch.lower()]) if ch.lower() in QWERTY and random.random()<rate else ch)
    return "".join(out)

def homoglyph_noise(s, rate=0.02):
    return "".join(HOMO.get(ch, ch) if random.random()<rate else ch for ch in s)

def ocr_noise(s, rate=0.01):
    for k,v in OCR_MAP.items():
        if random.random() < rate:
            s = s.replace(k, v)
    return s

def drop_punct_lower(s, drop=0.2):
    s = s.lower()
    return re.sub(r"[^\w\s]", lambda m: "" if random.random()<drop else m.group(0), s)

def truncate_text(s, keep_ratio=0.8):
    n = max(1, int(len(s)*keep_ratio))
    return s[:n]

def apply_corruption(text, severity=2):
    rate = 0.01*severity
    drop = 0.05*severity
    keep = 1.0 - 0.1*severity
    ops=[]
    if random.random()<0.7: ops.append(lambda x: keyboard_typo(x, rate))
    if random.random()<0.5: ops.append(lambda x: homoglyph_noise(x, rate))
    if random.random()<0.5: ops.append(lambda x: ocr_noise(x, rate))
    if random.random()<0.4: ops.append(lambda x: drop_punct_lower(x, drop))
    if random.random()<0.3: ops.append(lambda x: truncate_text(x, keep))
    random.shuffle(ops)
    for f in ops:
        text = f(text)
    return text


def _safe_get_text(item):
    if "raw_text" in item: return item["raw_text"]
    if "text" in item:     return item["text"]
    raise KeyError(f"no raw_text/text in item: keys={list(item.keys())}")

@dataclass
class BYOLNoiseCollator:
    tokenizer: BertTokenizer
    max_length: int = 128
    noise_prob: float = 1.0
    severity_schedule: List[int] = None
    pair_mode: str = "clean-noisy"
    _cur_epoch: int = 0

    def set_epoch(self, epoch:int): self._cur_epoch = epoch

    @property
    def cur_sev(self):
        if self.severity_schedule is None: return 2
        return self.severity_schedule[min(self._cur_epoch, len(self.severity_schedule)-1)]

    def _tok(self, texts: List[str]):
        return self.tokenizer(texts, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors='pt')

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        raw = [_safe_get_text(f) for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        view1 = raw
        if self.pair_mode == "clean-noisy":
            view2 = [apply_corruption(t, severity=self.cur_sev) if random.random()<self.noise_prob else t for t in raw]
        else:
            view2 = [apply_corruption(apply_corruption(t, self.cur_sev), self.cur_sev) for t in raw]

        enc1 = self._tok(view1)
        enc2 = self._tok(view2)

        batch = {
            "v1_input_ids": enc1["input_ids"],
            "v1_attention_mask": enc1["attention_mask"],
            "labels": labels,
            "v2_input_ids": enc2["input_ids"],
            "v2_attention_mask": enc2["attention_mask"],
        }
        if "token_type_ids" in enc1: batch["v1_token_type_ids"] = enc1["token_type_ids"]
        if "token_type_ids" in enc2: batch["v2_token_type_ids"] = enc2["token_type_ids"]
        return batch


class BYOLHead(nn.Module):
    def __init__(self, in_dim=768, proj_dim=256, hidden=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim)
        )
    def forward_online(self, x):
        z = self.projector(x); p = self.predictor(z)
        return F.normalize(p, dim=-1), F.normalize(z, dim=-1)
    def forward_target(self, x):
        z = self.projector(x)
        return F.normalize(z, dim=-1)

@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=(1.0 - tau))

class BYOLNoiseTrainer(Trainer):
    def __init__(self, *args, byol_lambda=1.0, ema_tau=0.996, **kwargs):
        super().__init__(*args, **kwargs)
        self.byol_lambda = byol_lambda
        self.ema_tau = ema_tau

        base = self.model.module if hasattr(self.model, "module") else self.model
        self.online_encoder = base.bert
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters(): p.requires_grad_(False)

        self.byol_head_online = BYOLHead(in_dim=768, proj_dim=256, hidden=512)
        self.byol_head_target = BYOLHead(in_dim=768, proj_dim=256, hidden=512)
        for p in self.byol_head_target.parameters(): p.requires_grad_(False)

        self.byol_head_online.to(self.args.device)
        self.byol_head_target.to(self.args.device)
        self.target_encoder.to(self.args.device)

    def create_optimizer(self):
        if self.optimizer is not None: return self.optimizer
        decay_params, no_decay_params = [], []
        def add_params(named_params):
            for n,p in named_params:
                if not p.requires_grad: continue
                if any(nd in n for nd in ["bias","LayerNorm.weight","layer_norm.weight","bn","BatchNorm"]):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
        add_params(self.model.named_parameters())
        add_params(self.byol_head_online.named_parameters())
        groups = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        opt_cls, opt_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = opt_cls(groups, **opt_kwargs)
        return self.optimizer

    @torch.no_grad()
    def _ema_update(self):
        ema_update(self.target_encoder, self.online_encoder, self.ema_tau)
        ema_update(self.byol_head_target, self.byol_head_online, self.ema_tau)

    def byol_loss(self, p_online, z_target):
        return 2 - 2 * (p_online * z_target.detach()).sum(dim=-1).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not model.training:
            outputs = model(
                input_ids=inputs["v1_input_ids"],
                attention_mask=inputs["v1_attention_mask"],
                token_type_ids=inputs.get("v1_token_type_ids"),
                labels=inputs["labels"],
            )
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        outputs_cls = model(
            input_ids=inputs["v1_input_ids"],
            attention_mask=inputs["v1_attention_mask"],
            token_type_ids=inputs.get("v1_token_type_ids"),
            labels=inputs["labels"],
        )
        loss_cls = outputs_cls.loss
        if loss_cls.dim() > 0: loss_cls = loss_cls.mean()

        v1 = self.online_encoder(
            input_ids=inputs["v1_input_ids"],
            attention_mask=inputs["v1_attention_mask"],
            token_type_ids=inputs.get("v1_token_type_ids"),
            return_dict=True,
        ).pooler_output
        with torch.no_grad():
            v2_t = self.target_encoder(
                input_ids=inputs["v2_input_ids"],
                attention_mask=inputs["v2_attention_mask"],
                token_type_ids=inputs.get("v2_token_type_ids"),
                return_dict=True,
            ).pooler_output
        p1, _ = self.byol_head_online.forward_online(v1)
        z2 = self.byol_head_target.forward_target(v2_t)
        loss_12 = self.byol_loss(p1, z2)

        v2 = self.online_encoder(
            input_ids=inputs["v2_input_ids"],
            attention_mask=inputs["v2_attention_mask"],
            token_type_ids=inputs.get("v2_token_type_ids"),
            return_dict=True,
        ).pooler_output
        with torch.no_grad():
            v1_t = self.target_encoder(
                input_ids=inputs["v1_input_ids"],
                attention_mask=inputs["v1_attention_mask"],
                token_type_ids=inputs.get("v1_token_type_ids"),
                return_dict=True,
            ).pooler_output
        p2, _ = self.byol_head_online.forward_online(v2)
        z1_t = self.byol_head_target.forward_target(v1_t)
        loss_21 = self.byol_loss(p2, z1_t)

        loss = loss_cls + self.byol_lambda * 0.5 * (loss_12 + loss_21)
        self._ema_update()
        if return_outputs: return loss, outputs_cls
        return loss
