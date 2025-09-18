
import argparse
import json
import torch
import wfdb
import numpy as np
from conversation import conv_vicuna_v1, PromptMode
import os
from transformers import AutoTokenizer
import pandas as pd
import sys
import time

sys.path.append("/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava")
sys.path.append('/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/eval')
from report.ecg_transform import PreprocessCfg, ecg_transform_v2

# 引入模型相关定义
from llava.model.multimodal_encoder.ked_encoder import xresnet1d101
from llava.model.multimodal_encoder.ked_head import TQNModel
from llava.model.multimodal_encoder.text_encoder import CLP_clinical

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

def predict_ecg_labels(ecg_tensor, ecg_model, text_encoder, tqn_model, label_features, label_texts, device):
    with torch.no_grad():
        input_tensor = ecg_tensor.unsqueeze(0).to(device)
        feats = ecg_model(input_tensor)
        feats = feats.transpose(1,2)
        out = tqn_model(feats, label_features)
        if out.shape[-1] == 2:
            probs = torch.softmax(out, dim=-1)[0]  # [num_labels, 2]
            pred = (probs[:, 1] > 0.3).long()
        else:
            probs = torch.sigmoid(out)[0]
            pred = (probs > 0.3).long()

    result_labels = [label for label, v in zip(label_texts, pred.cpu().numpy()) if v == 1]
    return result_labels

def load_ecg_signal(ecg_path):
    """加载单个ECG信号，支持 WFDB 格式，输出 shape [12, 5000]"""
    data = wfdb.rdsamp(ecg_path)[0]
    if 'mimic-iv' in ecg_path:
        data[[4, 5], :] = data[[5, 4], :]
    if data.shape[1] == 12:
        return torch.from_numpy(data.T).float()
    else:
        raise ValueError("ECG信号通道数非12道！")

def build_prompt_with_ecg(user_instruction, ecg_path, prelabels=None, image_process_mode="Default"):
    conv = conv_vicuna_v1.copy()
    conv.prompt_mode = PromptMode.DDP
    conv.ddp_labels = prelabels 
    conv.append_message("USER", (user_instruction, ecg_path, image_process_mode))
    conv.append_message("ASSISTANT", "")
    return conv.get_prompt()

def get_ecg_list(input_value):
    """根据输入类型返回ecg路径列表"""
    df = pd.read_csv(input_value)
    col = "path" 
    ecg_list = df[col].tolist()
    return ecg_list

def load_done_ecg(output_path):
    """读取已完成的ecg_path集合和已完成结果"""
    done_ecg = set()
    results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                for r in results:
                    if "ecg_path" in r:
                        done_ecg.add(r["ecg_path"])
            except Exception:
                # 如果不是标准JSON数组，尝试jsonlines读取
                f.seek(0)
                for line in f:
                    try:
                        r = json.loads(line)
                        if "ecg_path" in r:
                            done_ecg.add(r["ecg_path"])
                            results.append(r)
                    except:
                        continue
    return done_ecg, results

def main():
    parser = argparse.ArgumentParser(description="批量ECG信号推理脚本")
    parser.add_argument('--input', type=str,default='/data_C/sdb1/lyi/ECG-Chat-master/llava/llava/eval/report/ecg_test.csv', help="ECG信号列表文件或目录路径")
    parser.add_argument('--ckpt_path', type=str, default="/data_C/sdb1/lyi/ked3/control-spiderman-ECGFM-KED-456810e/trained_model/checkpoints_mimiciv/best_valid_all_increase_zhipuai_augment_epoch_7.pt")
    parser.add_argument('--bert_model_name', type=str, default="/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/model/multimodal_encoder/emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument('--label_csv', type=str, default="/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/all_label_standard.csv")
    parser.add_argument('--label_column', type=str, default="label")
    parser.add_argument('--tqn_layers', type=int, default=7)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--instruction", type=str, help="用户问题指令", default='Could you please help me explain my ECG? Do not explain what an ECG is, do not give background. Match the style of the reference reports, for example:\nYour ECG is normal. The heart’s rhythm and electrical conduction are within expected limits, and no abnormalities are detected.\nAlterations in the T wave (such as inversion, flattening, or peaking) are observed. This can be caused by myocardial ischemia, electrolyte imbalance, or other cardiac conditions. Further assessment may be needed if you experience symptoms.')
    parser.add_argument("--output", type=str, default='/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/eval/report/batch_result.json', help="输出JSON路径")
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args()

    preprocess_cfg = PreprocessCfg(seq_length=5000, duration=10, sampling_rate=500)
    test_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=False)

    # 断点续跑相关
    done_ecg, previous_results = load_done_ecg(args.output)
    ecg_list_all = get_ecg_list(args.input)
    ecg_list = [p for p in ecg_list_all if p not in done_ecg]
    print(f"共检测到{len(ecg_list_all)}条ECG信号，剩余待处理{len(ecg_list)}条...")

    # 1. 只加载一次ked
    ecg_model, text_encoder, tqn_model = load_cloud_model(
        args.ckpt_path, args.bert_model_name, args.device, tqn_layers=args.tqn_layers
    )
    ecg_model.eval()
    text_encoder.eval()
    tqn_model.eval()

    # 2. 只加载一次标签特征
    df = pd.read_csv(args.label_csv)
    label_texts = df[args.label_column].tolist()
    tokenizer_bert = AutoTokenizer.from_pretrained(args.bert_model_name)
    label_features = get_label_features(text_encoder, tokenizer_bert, label_texts, args.device)

    # 加载模型ecg-chat
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    base_model_dir = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/vicuna-13b-v1.5"
    lora_model_dir = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/checkpoints_1/llava-vicuna-13b-v1.5-finetune_lora"
    lora_model_dir_p = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/checkpoints_1/llava-vicuna-13b-v1.5-pretrain"
    print("Loading base model...")
    model = LlavaLlamaForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto")
    print("Base model loaded on:", next(model.parameters()).device)
    # 加载lora权重
    safetensors_path = os.path.join(lora_model_dir, "adapter_model.safetensors")
    from safetensors.torch import load_file as safetensors_load
    lora_weights = safetensors_load(safetensors_path)
    model.load_state_dict(lora_weights, strict=False)
    print("LoRA weights loaded.")

    non_lora = os.path.join(lora_model_dir, "non_lora_trainables.bin")
    if os.path.exists(non_lora):
        print(f"Loading non-LoRA weights from {non_lora} ...")
        non_lora_state = torch.load(non_lora, map_location=model.device)
        model.load_state_dict(non_lora_state, strict=False)
        print("Non-LoRA weights loaded.")
    projector_path = os.path.join(lora_model_dir_p, "mm_projector.bin")
    if os.path.exists(projector_path):
        projector_weights = torch.load(projector_path, map_location=model.device)
        model.load_state_dict(projector_weights, strict=False)
        print("Projector weights loaded.")
    model.eval()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=False)
    tokenizer.add_tokens(['<ecg>'], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    print("Model set to eval.")

    # 用于保存所有结果（已处理+新处理）
    all_results = previous_results.copy()

    # 主循环，每条实时追加写入
    with open(args.output, "a", encoding="utf-8") as f:
        for idx, ecg_path in enumerate(ecg_list):
            print(f"[{idx+1}/{len(ecg_list)}] 推理：{ecg_path}")
            try:
                ecg_tensor = load_ecg_signal(ecg_path)
                ecg_tensor = test_transform(ecg_tensor.unsqueeze(0)).squeeze(0)
                #=====生成标签============
                prelabels = predict_ecg_labels(
                    ecg_tensor,
                    ecg_model, text_encoder, tqn_model,
                    label_features, label_texts,
                    device=args.device
                )
                prompt = build_prompt_with_ecg(args.instruction, ecg_path, prelabels)
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                prompt_len = inputs["input_ids"].shape[1]
                ecg_tensor_for_model = ecg_tensor.unsqueeze(0).to(model.device)
                start_time = time.time()
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        ecgs=ecg_tensor,  # 按实际模型API调整
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                end_time = time.time()
                inference_time = end_time - start_time
                generated_ids = output_ids[0][prompt_len:]
                answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()              
                result = {
                    "ecg_path": ecg_path,
                    "prompt":prompt,
                    "prelabels": prelabels,
                    "model_output": answer,
                    "inference_time": inference_time
                }
                print(f"推理完成, 耗时{inference_time:.2f}秒")
            except Exception as e:
                print(f"推理失败: {ecg_path}, 错误: {e}")
                result = {
                    "ecg_path": ecg_path,
                    "error": str(e)
                }
            # 实时写入jsonlines
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            all_results.append(result)

    print(f"全部推理完成，结果已追加到 {args.output}")

    # （可选）最终合并为标准JSON数组文件
    with open(args.output.replace(".json", "_array.json"), "w", encoding="utf-8") as fout:
        json.dump(all_results, fout, ensure_ascii=False, indent=2)
    print(f"已保存为标准JSON数组文件：{args.output.replace('.json', '_array.json')}")

if __name__ == '__main__':
    main()