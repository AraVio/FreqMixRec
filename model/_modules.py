import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()
        hidden_size = args.hidden_size
        inner_size = 4 * args.hidden_size

        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(args.hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class EfficientDCT(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        N = x.shape[-1]
        k = torch.arange(N, device=x.device)[None, :] * np.pi / (2 * N)
        weights = torch.exp(-1j * k)
        
        x = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
        X = torch.fft.fft(x, dim=-1)
        
        V = 2 * (X * weights).real
        return V

class FrequencyFilterBlock(nn.Module):
    def __init__(self, hidden_size, dropout=0.2, use_context=True, scale=0.02):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_context = use_context

        self.plain_r = nn.Parameter(scale * torch.randn(1, 1, hidden_size))
        self.plain_i = nn.Parameter(scale * torch.randn(1, 1, hidden_size))
        self.plain_rb = nn.Parameter(scale * torch.randn(hidden_size))
        self.plain_ib = nn.Parameter(scale * torch.randn(hidden_size))

        if use_context:
            self.context_r = nn.Parameter(scale * torch.randn(1, 1, hidden_size))
            self.context_i = nn.Parameter(scale * torch.randn(1, 1, hidden_size))
            self.context_gate = nn.Sequential(
                nn.Linear(hidden_size, 4*hidden_size),
                nn.GELU(),
                nn.Linear(4*hidden_size, 2*hidden_size),
                nn.Sigmoid()
            )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def plain_filter(self, x_freq):
        real = F.relu(
            x_freq.real * self.plain_r - x_freq.imag * self.plain_i + self.plain_rb
        )
        imag = F.relu(
            x_freq.imag * self.plain_r + x_freq.real * self.plain_i + self.plain_ib
        )
        return torch.complex(real, imag)

    def context_filter(self, x_freq, x_time):
        gate = self.context_gate(x_time.mean(dim=1))
        rb_gate, ib_gate = gate.chunk(2, dim=-1)
        
        real = F.gelu(
            x_freq.real * self.context_r - x_freq.imag * self.context_i + 
            rb_gate.unsqueeze(1)
        )
        imag = F.gelu(
            x_freq.imag * self.context_r + x_freq.real * self.context_i + 
            ib_gate.unsqueeze(1)
        )
        return torch.complex(real, imag)

    def forward(self, x):
        identity = x
        
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')
        x_plain = self.plain_filter(x_freq)
        x = torch.fft.irfft(x_plain, n=x.size(1), dim=1, norm='ortho')
        x = self.norm1(x + identity)

        if self.use_context:
            x_freq_ctx = torch.fft.rfft(x, dim=1, norm='ortho')
            x_context = self.context_filter(x_freq_ctx, identity)
            x_filtered = torch.fft.irfft(x_context, n=x.size(1), dim=1, norm='ortho')
            x = x + x_filtered

        x = self.output_projection(x)
        return self.norm2(x + identity)

class MultiScaleDCT(nn.Module):
    def __init__(self, hidden_size, num_scales=3, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_scales = num_scales
        
        self.dct = EfficientDCT()
        
        self.freq_bands = self._compute_frequency_bands()
        
        self.freq_filters = nn.ModuleList([
            FrequencyFilterBlock(
                hidden_size=hidden_size,
                dropout=dropout,
                use_context=True
            )
            for _ in range(num_scales)
        ])
        
        self.freq_attention = nn.Sequential(
            nn.Linear(num_scales * hidden_size, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Sigmoid()
        )
        
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def _compute_frequency_bands(self):
        bands = []
        total_freq = self.hidden_size
        
        exp_scale = torch.exp(torch.linspace(0, torch.log(torch.tensor(total_freq)), 
                                           self.num_scales + 1))
        boundaries = (exp_scale * (total_freq / exp_scale[-1])).long()
        
        for i in range(self.num_scales):
            start_idx = boundaries[i].item()
            end_idx = boundaries[i + 1].item()
            bands.append((start_idx, end_idx))
        
        return bands

    def _create_smooth_mask(self, start_idx, end_idx, total_size, device):
        mask = torch.zeros(total_size, device=device)
        
        transition_width = min(
            max(2, (end_idx - start_idx) // 10),
            max(2, min(start_idx, total_size - end_idx) // 2)
        )
        
        mask[start_idx:end_idx] = 1.0
        
        if start_idx > 0:
            trans_start = max(0, start_idx - transition_width)
            length = start_idx - trans_start
            if length > 0:
                ramp_up = torch.linspace(0, 1, length, device=device)
                mask[trans_start:start_idx] = ramp_up

        if end_idx < total_size:
            trans_end = min(total_size, end_idx + transition_width)
            length = trans_end - end_idx
            if length > 0:
                ramp_down = torch.linspace(1, 0, length, device=device)
                mask[end_idx:trans_end] = ramp_down
        
        return mask

    def forward(self, x):
        residual = x
        
        x_freq = self.dct(x)
        
        multi_scale_features = []
        for scale, (start_idx, end_idx) in enumerate(self.freq_bands):
            mask = self._create_smooth_mask(
                start_idx, end_idx, self.hidden_size, x.device
            ).unsqueeze(0).unsqueeze(0)
            
            masked_freq = x_freq * mask
            
            scale_feature = self.freq_filters[scale](masked_freq)
            multi_scale_features.append(scale_feature)
        
        concat_features = torch.cat([f.mean(dim=1) for f in multi_scale_features], dim=-1)
        freq_weights = self.freq_attention(concat_features).unsqueeze(1)
        
        output = torch.stack(multi_scale_features, dim=-1)
        output = (output * freq_weights.unsqueeze(-1)).sum(dim=-1)
        output = self.dropout(output)
        
        output = output + residual
        output = self.output_norm(output)
        
        return output

class DCTBackbone(nn.Module):
    def __init__(self, hidden_size, num_scales=3, num_layers=1, dropout=0.5):
        super().__init__()
        layer = MultiScaleDCT(
            hidden_size=hidden_size,
            num_scales=num_scales,
            dropout=dropout
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FREQMIXBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dct_block = DCTBackbone(
            hidden_size=args.hidden_size,
            dropout=args.hidden_dropout_prob
        )
    
    def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=False):
        output = self.dct_block(hidden_states)
        
        if output_all_encoded_layers:
            return [hidden_states, output]
        else:
            return output