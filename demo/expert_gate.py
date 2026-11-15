"""One real expert compatibility gate. Does not load the complete model."""
import argparse
import gc
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True, help='Immutable Hugging Face model commit SHA')
    parser.add_argument('--output', default='expert-gate.json')
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        parser.error('--revision must be a 40-character commit SHA')
    import torch
    import transformers
    import bitsandbytes as bnb
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
    if transformers.__version__ != '4.57.1' or not torch.cuda.is_available():
        raise RuntimeError('Requires Transformers 4.57.1 and CUDA')
    model_id = 'allenai/OLMoE-1B-7B-0924'
    config = AutoConfig.from_pretrained(model_id, revision=args.revision)
    if config.model_type != 'olmoe':
        raise RuntimeError('Expected OLMoE architecture')
    index = json.loads(Path(hf_hub_download(model_id, 'model.safetensors.index.json', revision=args.revision)).read_text())
    expert = OlmoeMLP(config).half().eval()
    # Read only expert zero's three projections into CPU memory.
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        key = f'model.layers.0.mlp.experts.0.{name}.weight'
        shard = hf_hub_download(model_id, index['weight_map'][key], revision=args.revision)
        with safe_open(shard, framework='pt', device='cpu') as f:
            getattr(expert, name).weight.data.copy_(f.get_tensor(key))
    x = torch.randn(8, config.hidden_size, device='cuda', dtype=torch.float16)
    with torch.inference_mode():
        expert.cuda()
        reference = expert(x).cpu()
        expert.cpu()
    records = []
    for tier in ('fp16', 'int8', 'nf4'):
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        candidate = OlmoeMLP(config).half().eval()
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            source = getattr(expert, name)
            if tier == 'fp16':
                linear = torch.nn.Linear(source.in_features, source.out_features, bias=False, dtype=torch.float16)
            elif tier == 'int8':
                linear = bnb.nn.Linear8bitLt(source.in_features, source.out_features, bias=False, has_fp16_weights=False)
            else:
                linear = bnb.nn.Linear4bit(source.in_features, source.out_features, bias=False, quant_type='nf4', compute_dtype=torch.float16)
            linear.load_state_dict(source.state_dict())
            setattr(candidate, name, linear.to('cuda'))
        candidate.cuda()
        with torch.inference_mode():
            y = candidate(x)
        torch.cuda.synchronize()
        if not torch.isfinite(y).all():
            raise RuntimeError(f'{tier}: non-finite output')
        dtypes = {name: str(getattr(candidate, name).weight.dtype) for name in ('gate_proj', 'up_proj', 'down_proj')}
        expected = {'fp16': torch.float16, 'int8': torch.int8, 'nf4': torch.uint8}[tier]
        if any(getattr(candidate, name).weight.dtype != expected for name in dtypes):
            raise RuntimeError(f'{tier}: packed storage check failed: {dtypes}')
        if tier == 'fp16':
            torch.testing.assert_close(y.detach().cpu(), reference, rtol=1e-3, atol=1e-3)
        error = (y.detach().cpu().float() - reference.float()).abs().max().item()
        records.append({'tier': tier, 'dtypes': dtypes, 'allocated_bytes_above_baseline': torch.cuda.memory_allocated() - baseline,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'max_abs_error_vs_fp16': error})
        del y, candidate, linear
    sizes = [r['allocated_bytes_above_baseline'] for r in records]
    if not sizes[0] > sizes[1] > sizes[2]:
        raise RuntimeError(f'Expected FP16 > INT8 > NF4 live allocations, got {sizes}')
    result = {'scope': 'one real expert only; not a full-model budget certification', 'model_id': model_id,
              'revision': args.revision, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'transformers': transformers.__version__, 'bitsandbytes': bnb.__version__, 'results': records}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
