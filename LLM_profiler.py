

import os as _os
_os.environ["HUGGINGFACE_HUB_TOKEN"] = "token"

import os
import pickle
import sys
import math
from typing import Tuple, Dict

import numpy as np
import torch
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig


MODEL_NAMES = [
    "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
    "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
    "mistralai/Mistral-7B-v0.1"
]
SHAPES_TO_PROFILE = [
    (4, 256),
]
OUTPUT_DIR = "profiles"



STORAGE_DW = 4        # bytes per parameter (fp32)
PRECISION = "fp16"    # or "fp32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SUPPORTED = {
    'MultiheadAttention', 'GPT2Attention', 'LlamaAttention', 'FalconAttention',
    'GemmaAttention', 'MistralAttention', 'MistralAttentionFlash',
    'MixtureOfExpertsBlock',
    'GPT2MLP', 'LlamaMLP', 'FalconMLP', 'SwiGLU',
    'LayerNorm', 'RMSNorm', 'LlamaRMSNorm', 'T5LayerNorm',
    'Embedding', 'Dropout',
    'RotaryEmbedding', 'LlamaSiLU',
    'MistralBlock', 'MistralDecoderLayer'
}

def add_profile_entry(meta: Dict[int, Dict], mid: int, **kwargs):

    if mid not in meta:
        meta[mid] = kwargs

def analyze_model_workload(
        model,
        shape: Tuple[int, int],
        use_gradient_checkpointing: bool = False,
):
    batch_size, seq_len = shape
    act_dw = 2 if PRECISION == 'fp16' else 4
    shape_layer_meta: Dict[int, Dict] = {}

    # total parameter & optimizer bytes
    total_param_bytes = sum(p.numel() * STORAGE_DW for p in model.parameters())
    total_opt_bytes   = 2 * total_param_bytes  # assume Adam-like state

    #batch_size, seq_len = shape


    def fwd_hook(m, inp, out):
        # extract output tensor y
        if hasattr(out, 'last_hidden_state') and isinstance(out.last_hidden_state, torch.Tensor):
            y = out.last_hidden_state
        elif isinstance(out, torch.Tensor):
            y = out
        elif isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
            y = out[0]
        else:
            return

        # extract input tensor x
        if isinstance(inp, (list, tuple)) and inp and isinstance(inp[0], torch.Tensor):
            x = inp[0]
        elif isinstance(inp, torch.Tensor):
            x = inp
        else:
            return

        nm = m.__class__.__name__
        if nm not in SUPPORTED:
            return

        mid = id(m)
        p_bytes   = sum(p.numel() * STORAGE_DW for p in m.parameters())
        act_in    = x.numel() * act_dw
        act_out   = y.numel() * act_dw

        B, S = x.shape[0], x.shape[1]
        E    = getattr(model.config, 'hidden_size', x.shape[-1])
        H    = max(getattr(model.config, 'num_attention_heads', 1), 1)


        if 'MixtureOfExpertsBlock' in nm:
            k = getattr(model.config, 'num_experts_per_tok', 2)
            ops_fwd = k * (8 * B * S * E * E)
        elif 'MistralAttention' in nm:
            # grouped-query / sliding-window attention
            W = getattr(model.config, 'sliding_window', S)
            ops_fwd = (4 * B * S * E * E) + (2 * B * H * S * W * (E // H))
        elif 'Attention' in nm:
            ops_fwd = (4 * B * S * E * E) + (2 * B * H * S * S * (E // H))
        elif 'SwiGLU' in nm:
            # feed-forward with gated activation
            ops_fwd = 8 * B * S * E * E
        elif 'MLP' in nm:
            ops_fwd = 8 * B * S * E * E
        elif 'Norm' in nm:
            ops_fwd = 5 * x.numel()
        else:
            ops_fwd = 0

        dropout_buf  = act_out / 8 if 'dropout' in nm.lower() else 0
        bytes_mv_fwd = act_in + act_out + dropout_buf
        ws_fwd       = p_bytes + act_in + act_out + dropout_buf

        add_profile_entry(
            shape_layer_meta, mid,
            ops_fwd=ops_fwd, bytes_mv_fwd=bytes_mv_fwd, ws_fwd=ws_fwd,
            param_bytes=p_bytes, act_in_bytes=act_in, out_act_bytes=act_out,
            name=nm, seq_scale_power=2 if 'Attention' in nm else 1
        )


        y.register_hook(lambda grad, m_id=mid: bwd_cb(m_id, grad))


    def bwd_cb(mid: int, grad):
        if mid not in shape_layer_meta or 'ops_bwd' in shape_layer_meta[mid]:
            return
        meta = shape_layer_meta[mid]
        nm   = meta['name']

        # choose backward factor by layer type
        if 'MistralAttention' in nm:
            factor = 3.0
        elif 'SwiGLU' in nm:
            factor = 2.2
        elif any(k in nm for k in ['MixtureOfExpertsBlock']):
            factor = 2.0
        elif 'Attention' in nm:
            factor = 2.5
        elif 'Norm' in nm:
            factor = 1.5
        else:
            factor = 1.0

        ops_fwd = meta['ops_fwd']
        ops_bwd = factor * ops_fwd
        if use_gradient_checkpointing:
            ops_bwd += ops_fwd


        grad_bytes = grad.numel() * act_dw if grad is not None else 0
        param_bytes = meta['param_bytes']

        shape_layer_meta[mid].update({
            'ops_bwd': ops_bwd,
            'bytes_mv_bwd': param_bytes + grad_bytes,
            'ws_bwd': param_bytes
                      + meta['act_in_bytes']
                      + meta['out_act_bytes']
                      + param_bytes
                      + grad_bytes
        })


    hooks = [m.register_forward_hook(fwd_hook) for m in model.modules()]

    # run one train step
    model.train()
    inp = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_len), device=device
    )
    out = model(inp)
    model.zero_grad(set_to_none=True)
    loss = (
        out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
    ).sum()
    loss.backward()


    for h in hooks:
        h.remove()


    layer_profiles = list(shape_layer_meta.values())
    ops_total   = np.log1p([p.get('ops_fwd', 0) + p.get('ops_bwd', 0) for p in layer_profiles])
    bytes_total = np.log1p([p.get('bytes_mv_fwd', 0) + p.get('bytes_mv_bwd', 0) for p in layer_profiles])
    norm_stats = {
        'ops_total_mean': ops_total.mean(),
        'ops_total_std': max(ops_total.std(), 1e-6),
        'bytes_total_mean': bytes_total.mean(),
        'bytes_total_std': max(bytes_total.std(), 1e-6),
    }
    for key in ['ops_fwd', 'bytes_mv_fwd', 'ws_fwd', 'ops_bwd', 'bytes_mv_bwd', 'ws_bwd']:
        vals = np.array([p.get(key, 0) for p in layer_profiles], dtype=np.float64)
        logv = np.log1p(vals)
        norm_stats[key] = {'mean': logv.mean(), 'std': max(logv.std(), 1e-6)}

    return {
        'layers': layer_profiles,
        'norm_stats': norm_stats,
        'global_opt_bytes': total_opt_bytes
    }


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name in MODEL_NAMES:
        for bs, sl in SHAPES_TO_PROFILE:
            print(f"\n→ Profiling {model_name} @ batch={bs}, seq_len={sl}")
            try:
                config = AutoConfig.from_pretrained(model_name)
                bnb_cfg = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    config=config,
                    device_map="auto",
                    quantization_config=bnb_cfg,
                    low_cpu_mem_usage=True,
                    offload_folder="offload",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    use_auth_token=True
                )
            except Exception as e:
                print(f"  ✖ Failed to load {model_name}: {e}")
                continue


            if getattr(model, "num_parameters", lambda: 0)() > 2e9 and device.type == 'cuda':
                model.gradient_checkpointing_enable()

            if not getattr(model, "is_loaded_in_8bit", False):
                model.to(device)
            if not hasattr(model, 'config'):
                model.config = config

            profiling_results = analyze_model_workload(
                model,
                shape=(bs, sl),
                use_gradient_checkpointing=getattr(model, "gradient_checkpointing", False)
            )

            safe_name = model_name.replace("/", "_")
            out_name = f"{safe_name}_bs{bs}_sl{sl}.pkl"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, 'wb') as f:
                pickle.dump(profiling_results, f)

            print(f"  ✔ Saved profile ({len(profiling_results['layers'])} layers) to {out_path}")
