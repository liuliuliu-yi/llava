"""
Usage:
python3 -m fastchat.model.apply_delta --base ~/model_weights/llama-7b --target ~/model_weights/vicuna-7b --delta lmsys/vicuna-7b-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava import LlavaLlamaForCausalLM


def apply_delta(base_model_path, target_model_path, delta_path):
    #加载基础模型（base）
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    #加载增量权重（delta）
    print("Loading delta")
    delta = LlavaLlamaForCausalLM.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path)
    #权重合成（Apply delta）
    print("Applying delta")
    for name, param in tqdm(delta.state_dict().items(), desc="Applying delta"):
        if name not in base.state_dict():
            # # 只允许新增的多模态头参数（如mm_projector），其他新参数要报错
            assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f'{name} not in base model'
            continue
        if param.data.shape == base.state_dict()[name].shape:
            ## 形状相同，直接加base的权重
            param.data += base.state_dict()[name]
        else:
            ## 形状不同（如词表扩展），只允许embed_tokens和lm_head
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], \
                f'{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}'
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] += bparam

    print("Saving target model")
    delta.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
