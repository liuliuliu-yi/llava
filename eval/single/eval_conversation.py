import argparse
import json
import torch
import wfdb
import numpy as np
import os
import sys
import time
from transformers import AutoTokenizer
import pandas as pd

sys.path.append("/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava")
sys.path.append('/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/eval')
from report.conversation import conv_vicuna_v1, PromptMode
from report.ecg_transform import PreprocessCfg, ecg_transform_v2
from llava.model.multimodal_encoder.ked_encoder import xresnet1d101
from llava.model.multimodal_encoder.ked_head import TQNModel
from llava.model.multimodal_encoder.text_encoder import CLP_clinical
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

def load_cloud_model(ckpt_path, bert_model_name, device, tqn_layers=7):
    checkpoint = torch.load(ckpt_path, map_location=device)
    ked_encoder = xresnet1d101(input_channels=12, kernel_size=5, use_ecgNet_Diagnosis='other').to(device)
    ked_encoder.load_state_dict(checkpoint['ecg_model'])
    ked_encoder.eval()
    text_encoder = CLP_clinical(bert_model_name=bert_model_name, embed_dim=768).to(device)
    text_encoder.load_state_dict(checkpoint['text_encoder'], strict=False)
    text_encoder.eval()
    tqn_model = TQNModel(embed_dim=768, num_layers=tqn_layers).to(device)
    tqn_model.load_state_dict(checkpoint['model'])
    tqn_model.eval()
    return ked_encoder, text_encoder, tqn_model

def get_label_features(text_encoder, tokenizer, label_texts, device):
    text_batch = tokenizer(label_texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        label_features = text_encoder.encode_text(text_batch)
    return label_features

def predict_ecg_labels(ecg_tensor, ckpt_path, bert_model_name, label_csv, label_column, tqn_layers, device):
    ecg_model, text_encoder, tqn_model = load_cloud_model(
        ckpt_path, bert_model_name, device, tqn_layers=tqn_layers
    )
    df = pd.read_csv(label_csv)
    label_texts = df[label_column].tolist()
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    label_features = get_label_features(text_encoder, tokenizer, label_texts, device)
    with torch.no_grad():
        input_tensor = ecg_tensor.unsqueeze(0).to(device)
        feats = ecg_model(input_tensor)
        feats = feats.transpose(1, 2)
        out = tqn_model(feats, label_features)
        if out.shape[-1] == 2:
            pred = torch.argmax(out, dim=-1)[0]
        else:
            pred = (torch.sigmoid(out)[0] > 0.5).long()
    result_labels = [label for label, v in zip(label_texts, pred.cpu().numpy()) if v == 1]
    return result_labels

def load_ecg_signal(ecg_path):
    data = wfdb.rdsamp(ecg_path)[0]
    if 'mimic-iv' in ecg_path:
        data[[4, 5], :] = data[[5, 4], :]
    if data.shape[1] == 12:
        return torch.from_numpy(data.T).float()
    else:
        raise ValueError("ECG信号通道数非12道！")

def build_prompt_with_ecg(conv, user_instruction, ecg_path, prelabels=None, image_process_mode="Default"):
    # 多轮对话时conv对象需复用，历史对话自动累加
    if prelabels:
        conv.prompt_mode = PromptMode.DDP
        conv.ddp_labels = prelabels
    conv.append_message("USER", (user_instruction, ecg_path, image_process_mode))
    conv.append_message("ASSISTANT", "")
    return conv.get_prompt()

def main():
    parser = argparse.ArgumentParser(description="ECG多轮对话推理脚本")
    parser.add_argument('--ckpt_path', type=str, default="/data_C/sdb1/lyi/ked3/control-spiderman-ECGFM-KED-456810e/trained_model/checkpoints_mimiciv/best_valid_all_increase_zhipuai_augment_epoch_7.pt")
    parser.add_argument('--bert_model_name', type=str, default="/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/model/multimodal_encoder/emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument('--label_csv', type=str, default="/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/all_label_standard.csv")
    parser.add_argument('--label_column', type=str, default="label")
    parser.add_argument('--tqn_layers', type=int, default=7)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ecg", type=str, default='/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/mimiciv/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/p1153/p11531320/s46743942/46743942', help="ECG信号文件路径")
    parser.add_argument("--output", type=str, default='./multiturn_result.json', help="输出JSON路径")
    args = parser.parse_args()

    preprocess_cfg = PreprocessCfg(seq_length=5000, duration=10, sampling_rate=500)
    test_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=False)
    ecg_tensor = load_ecg_signal(args.ecg)
    ecg_tensor = test_transform(ecg_tensor.unsqueeze(0)).squeeze(0)

    prelabels = predict_ecg_labels(
        ecg_tensor,
        ckpt_path=args.ckpt_path,
        bert_model_name=args.bert_model_name,
        label_csv=args.label_csv,
        label_column=args.label_column,
        tqn_layers=args.tqn_layers,
        device=args.device
    )

    # ====== LLaVA模型加载部分 =======
    base_model_dir = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/vicuna-13b-v1.5"
    lora_model_dir = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/checkpoints/llava-vicuna-13b-v1.5-finetune_lora"
    lora_model_dir_p = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/checkpoints/llava-vicuna-13b-v1.5-pretrain"

    model = LlavaLlamaForCausalLM.from_pretrained(
        base_model_dir, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto"
    )
    # 加载lora权重
    safetensors_path = os.path.join(lora_model_dir, "adapter_model.safetensors")
    from safetensors.torch import load_file as safetensors_load
    lora_weights = safetensors_load(safetensors_path)
    model.load_state_dict(lora_weights, strict=False)

    non_lora = os.path.join(lora_model_dir, "non_lora_trainables.bin")
    if os.path.exists(non_lora):
        non_lora_state = torch.load(non_lora, map_location=model.device)
        model.load_state_dict(non_lora_state, strict=False)
    projector_path = os.path.join(lora_model_dir_p, "mm_projector.bin")
    if os.path.exists(projector_path):
        projector_weights = torch.load(projector_path, map_location=model.device)
        model.load_state_dict(projector_weights, strict=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=False)

    # ====== 多轮对话部分 ======
    conv = conv_vicuna_v1.copy()
    if prelabels:
        conv.prompt_mode = PromptMode.DDP
        conv.ddp_labels = prelabels

    max_turns = 5
    result = {
        "ecg_shape": list(ecg_tensor.shape),
        "prelabels": prelabels,
        "turns": []
    }

    for turn in range(max_turns):
        user_instruction = input(f"第{turn+1}轮，请输入您的问题(输入exit退出): ").strip()
        if user_instruction.lower() == "exit":
            break
        prompt = build_prompt_with_ecg(conv, user_instruction, args.ecg, prelabels if turn==0 else None)
        print("=================================提示词===================================")
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        ecg_tensor_for_model = ecg_tensor.unsqueeze(0).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                ecgs=ecg_tensor,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][prompt_len:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"【AI回复】：{answer}\n")
        conv.append_message("ASSISTANT", answer)
        result["turns"].append({"user": user_instruction, "assistant": answer})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()