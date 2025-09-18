import argparse
import json
import torch
import wfdb
import numpy as np
from conversation import conv_vicuna_v1,PromptMode
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import pandas as pd
import sys
import time   # 新增
sys.path.append("/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava")
sys.path.append('/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/eval')
from report.ecg_transform import PreprocessCfg, ecg_transform_v2
# 你需要引入这两个类定义（可以根据实际路径import）
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



def predict_ecg_labels(ecg_tensor,ckpt_path,bert_model_name,label_csv,label_column,tqn_layers,device):
   


    # ========== 加载模型 ==========
    ecg_model, text_encoder, tqn_model = load_cloud_model(
        ckpt_path, bert_model_name,device, tqn_layers=tqn_layers
    )

    # ========== 准备标签特征 ==========
    df = pd.read_csv(label_csv)
    label_texts = df[label_column].tolist()
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    label_features = get_label_features(text_encoder, tokenizer, label_texts, device)

    # ========== 推理 ==========
    with torch.no_grad():
        input_tensor = ecg_tensor.unsqueeze(0).to(device)
        feats = ecg_model(input_tensor)
        feats = feats.transpose(1,2)
        out = tqn_model(feats, label_features)
        if out.shape[-1] == 2:
            probs = torch.softmax(out, dim=-1)[0]  # [num_labels, 2]
            print("标签阳性概率:", probs[:, 1].cpu().numpy())
            pred = (probs[:, 1] > 0.3).long()
        else:
            probs = torch.sigmoid(out)[0]
            print("标签概率:", probs.cpu().numpy())
            pred = (probs > 0.3).long()

    # ========== 输出结果 ==========
    # 打印每个标签的预测结果
    # label_texts 是所有标签名的列表
    result_labels = [label for label, v in zip(label_texts, pred.cpu().numpy()) if v == 1]
    
    # 输出所有属于的标签
    print(result_labels) #['non-specific ST segment junctional depression', 'sinus rhythm']

    return result_labels






def load_ecg_signal(ecg_path):
    """
    加载单个ECG信号，支持 WFDB 格式（通过wfdb），输出 shape 为 [12, 5000]
    """
    data = wfdb.rdsamp(ecg_path)[0]  # shape: (5000, 12)
    if 'mimic-iv' in ecg_path : 
            # 交换avL和avF列 #(12, 5000)
            data[[4, 5], :] = data[[5, 4], :]
    if data.shape[1] == 12:
        return torch.from_numpy(data.T).float()  # shape: [12, 5000]
    else:
        raise ValueError("ECG信号通道数非12道！")

def build_prompt_with_ecg(user_instruction, ecg_path,prelabels=None,image_process_mode="Default"):
    """
    使用conv_vicuna_v1模板，结合ECG信号特殊处理，生成prompt
    """
    conv = conv_vicuna_v1.copy()
    if prelabels:
        conv.prompt_mode = PromptMode.DDP
        conv.ddp_labels = prelabels
    # 按照特殊信号处理方式，user消息为元组：(文本, ecg路径, 处理方式)
    conv.append_message("USER", (user_instruction, ecg_path, image_process_mode))
    conv.append_message("ASSISTANT", "")
    return conv.get_prompt()


def main():
    parser = argparse.ArgumentParser(description="ECG单条信号推理脚本")
    parser.add_argument('--ckpt_path', type=str, default="/data_C/sdb1/lyi/ked3/control-spiderman-ECGFM-KED-456810e/trained_model/checkpoints_mimiciv/best_valid_all_increase_zhipuai_augment_epoch_7.pt")
    parser.add_argument('--bert_model_name', type=str, default="/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/model/multimodal_encoder/emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument('--label_csv', type=str, default="/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_1/dataset/all_label_standard.csv")
    parser.add_argument('--label_column', type=str, default="label")
    parser.add_argument('--tqn_layers', type=int, default=7)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ecg", type=str,default='/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/records500/13000/13339_hr',help="ECG信号文件路径(WFDB格式)")
    parser.add_argument("--instruction", type=str,help="用户问题指令",default='What should I know about my ECG results?')
    parser.add_argument("--output", type=str,default='/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/eval/single/result.json' , help="输出JSON路径")
    args = parser.parse_args()

    # ========== ECG预处理 ==========
    preprocess_cfg = PreprocessCfg(
        seq_length=5000,
        duration=10,
        sampling_rate=500,
    )
    test_transform = ecg_transform_v2(
        cfg=preprocess_cfg,
        is_train=False
    )

    # ========== 加载信号 ==========
    ecg_tensor = load_ecg_signal(args.ecg)
    ecg_tensor = test_transform(ecg_tensor.unsqueeze(0)).squeeze(0)  # [12, 5000]
    
    #=====生成标签============
    prelabels = predict_ecg_labels(ecg_tensor,ckpt_path=args.ckpt_path,bert_model_name=args.bert_model_name,label_csv=args.label_csv,label_column=args.label_column,tqn_layers=args.tqn_layers,device=args.device)
  
    # 生成带有<ecg>特殊标记的prompt
   
    prompt = build_prompt_with_ecg(args.instruction, args.ecg , prelabels)


    print(prompt)
    # 加载模型、推理（请根据实际模型API补全）
    # ====== LLaVA模型加载部分 =======
    # 确保llava源码路径在PYTHONPATH（如果已pip install -e .可省略）
    
    from transformers import AutoTokenizer
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    base_model_dir = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/vicuna-13b-v1.5"
    lora_model_dir = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/checkpoints_1/llava-vicuna-13b-v1.5-finetune_lora"
    lora_model_dir_p = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/checkpoints_1/llava-vicuna-13b-v1.5-pretrain"

    
    print("Loading base model...")
    model = LlavaLlamaForCausalLM.from_pretrained(base_model_dir,torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")
    print("Base model loaded on:", next(model.parameters()).device)
    
    # # 加载lora权重
    # lora_weights_path = os.path.join(lora_model_dir, "adapter_model.bin")
    # print(f"Loading LoRA weights from {lora_weights_path} ...")
    # lora_weights = torch.load(lora_weights_path, map_location=model.device)
    # model.load_state_dict(lora_weights, strict=False)
    # print("LoRA weights loaded.")

    # 加载lora权重
    bin_path = os.path.join(lora_model_dir, "adapter_model.bin")
    safetensors_path = os.path.join(lora_model_dir, "adapter_model.safetensors")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file as safetensors_load
        print(f"Loading LoRA weights from {safetensors_path} ...")
        lora_weights = safetensors_load(safetensors_path)
    elif os.path.exists(bin_path):
        print(f"Loading LoRA weights from {bin_path} ...")
        lora_weights = torch.load(bin_path, map_location=model.device)
    else:
        raise FileNotFoundError("No adapter_model.bin or adapter_model.safetensors found!")

    model.load_state_dict(lora_weights, strict=False)
    print("LoRA weights loaded.")

    # 可选加载non_lora_trainables
    non_lora = os.path.join(lora_model_dir, "non_lora_trainables.bin")
    if os.path.exists(non_lora):
        print(f"Loading non-LoRA weights from {non_lora} ...")
        non_lora_state = torch.load(non_lora, map_location=model.device)
        model.load_state_dict(non_lora_state, strict=False)
        print("Non-LoRA weights loaded.")

    # 新增：加载projector权重
    projector_path = os.path.join(lora_model_dir_p, "mm_projector.bin")
    if os.path.exists(projector_path):
        projector_weights = torch.load(projector_path, map_location=model.device)
        model.load_state_dict(projector_weights, strict=False)
        print("Projector weights loaded.")

    model.eval()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=False)


    print("Model set to eval.")
    # === 修正embedding indices must be tensor not NoneType ===
    print("Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    print("Prompt tokenized.")

    print("Preparing ECG tensor for model...")
    ecg_tensor_for_model = ecg_tensor.unsqueeze(0).to(model.device)
    print("ECG tensor for model shape:", ecg_tensor_for_model.shape)
    print("ECG tensor device:", ecg_tensor_for_model.device)

    start_time = time.time()   # 新增：记录开始时间
    print("Starting generation...")
    with torch.no_grad():
        #output = model(ecg_tensor.unsqueeze(0), prompt=prompt)
         output_ids = model.generate(
            input_ids=inputs["input_ids"],         # 重点：不是inputs=，而是input_ids=
            attention_mask=inputs.get("attention_mask", None),
            ecgs=ecg_tensor,                      # 你的ECG张量，模型会自动插入
            max_new_tokens=128,                   # 你期望生成的最大长度
            do_sample=False,                      # 是否采样/greedy
            num_beams=1,                          # beam search数量
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            )
    print("Generation finished.")
    end_time = time.time()     # 新增：记录结束时间
    inference_time = end_time - start_time   # 新增：计算推理时间
    print("推理耗时：{:.4f} 秒".format(inference_time))   # 新增：打印推理耗时


    generated_ids = output_ids[0][prompt_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()        
    # 输出（这里只输出prompt和信号shape，实际推理时请补充模型输出）
    # 输出（这里只输出prompt和信号shape及模型生成内容）
    result = {
        "prompt": prompt,
        "ecg_shape": list(ecg_tensor.shape),
        "prelabels": prelabels,
        "model_output": answer,
    }
    with open(args.output, "a", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    

if __name__ == '__main__':
    main()