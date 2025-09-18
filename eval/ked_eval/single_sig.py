# --coding:utf-8--

import sys
sys.path.append('/data_C/sdb1/lyi/ECG-Chat-master/llava')
sys.path.append('/data_C/sdb1/lyi/ECG-Chat-master')
import os
import torch
import numpy as np
import pandas as pd
from llava.model.multimodal_encoder.ked_encoder import KEDEncoderWrapper, xresnet1d101
from llava.model.multimodal_encoder.ked_head import TQNModel
from llava.model.multimodal_encoder.text_encoder import CLP_clinical
from transformers import AutoTokenizer
import wfdb
from edge_1.dataset.ecg_transform import PreprocessCfg, ecg_transform_v2
import argparse

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

def load_single_ecg_signal(ecg_path):
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

def main():
    parser = argparse.ArgumentParser(description="单个ECG信号标签推理脚本") 
    parser.add_argument('--ecg', type=str, default="/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/records500/17000/17420_hr", help='ECG信号文件路径（WFDB格式）')
    parser.add_argument('--ckpt_path', type=str, default="/data_C/sdb1/lyi/ked3/control-spiderman-ECGFM-KED-456810e/trained_model/checkpoints_mimiciv/best_valid_all_increase_zhipuai_augment_epoch_7.pt")
    parser.add_argument('--bert_model_name', type=str, default="/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/model/multimodal_encoder/emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument('--label_csv', type=str, default="/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/all_label_standard.csv")
    parser.add_argument('--label_column', type=str, default="label")
    parser.add_argument('--tqn_layers', type=int, default=7)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    ecg_tensor = load_single_ecg_signal(args.ecg)
    ecg_tensor = test_transform(ecg_tensor.unsqueeze(0)).squeeze(0)  # [12, 5000]

    # ========== 加载模型 ==========
    ecg_model, text_encoder, tqn_model = load_cloud_model(
        args.ckpt_path, args.bert_model_name, args.device, tqn_layers=args.tqn_layers
    )

    # ========== 准备标签特征 ==========
    df = pd.read_csv(args.label_csv)
    label_texts = df[args.label_column].tolist()
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    label_features = get_label_features(text_encoder, tokenizer, label_texts, args.device)

    # ========== 推理 ==========
    with torch.no_grad():
        input_tensor = ecg_tensor.unsqueeze(0).to(args.device)  # [1, 12, 5000]
        feats = ecg_model(input_tensor)      # [1, 768, seq_len]
        feats = feats.transpose(1,2)         # [1, seq_len, 768]
        out = tqn_model(feats, label_features)  # [1, num_labels, 2]
        if out.shape[-1] == 2:
            pred = torch.argmax(out, dim=-1)[0]  # [num_labels]
        else:
            pred = (torch.sigmoid(out)[0] > 0.5).long()  # [num_labels]

    # ========== 输出结果 ==========
    # 打印每个标签的预测结果
    # label_texts 是所有标签名的列表
    result_labels = [label for label, v in zip(label_texts, pred.cpu().numpy()) if v == 1]
    
    # 输出所有属于的标签
    print(result_labels) #['non-specific ST segment junctional depression', 'sinus rhythm']

if __name__ == "__main__":
    main()