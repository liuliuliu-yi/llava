# --coding:utf-8--

import sys
sys.path.append('/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava')
sys.path.append('/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master')
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from llava.model.multimodal_encoder.ked_encoder import KEDEncoderWrapper, xresnet1d101
from llava.model.multimodal_encoder.ked_head import TQNModel
from llava.model.multimodal_encoder.text_encoder import CLP_clinical
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import wfdb
from edge_2.dataset.ecg_transform import PreprocessCfg, ecg_transform_v2

# ==== 1. 自定义Dataset ====
class ECGSignalDataset(Dataset):
    """
    ECG信号数据集，支持npy或内存array。
    """
    def __init__(self, signals, transforms=None):
        self.signals = signals  # list
        self.transforms = transforms
        # 判断 self.signals[0] 是不是str，决定 use_path
        self.use_path = isinstance(self.signals[0], str)
        if self.use_path:
            import wfdb
            self.wfdb = wfdb

    def __len__(self):
        return len(self.signals)
    def __getitem__(self, idx):
        if self.use_path:
            path = self.signals[idx]
            sig, fields = self.wfdb.rdsamp(path)
        else:
            sig = self.signals[idx]
        if sig.shape != (12, 5000):
            sig = sig.T
        if isinstance(sig, np.ndarray):
            sig = torch.from_numpy(sig).float()
        if self.transforms is not None:
            sig = self.transforms(sig.unsqueeze(0)).squeeze(0)
        if torch.any(torch.isnan(sig)):
            sig = torch.where(torch.isnan(sig), torch.zeros_like(sig), sig)
        return sig

def load_cloud_model(ckpt_path, bert_model_name, device, tqn_layers=7):
    checkpoint = torch.load(ckpt_path, map_location=device)
    # ECG编码器
    ked_encoder = xresnet1d101(input_channels=12, kernel_size=5, use_ecgNet_Diagnosis='other').to(device)
    ked_encoder.load_state_dict(checkpoint['ecg_model'])
    ked_encoder.eval()
    # 文本编码器
    text_encoder = CLP_clinical(bert_model_name=bert_model_name, embed_dim=768).to(device)
    text_encoder.load_state_dict(checkpoint['text_encoder'], strict=False)
    text_encoder.eval()
    # TQN模型
    tqn_model = TQNModel(embed_dim=768, num_layers=tqn_layers).to(device)
    tqn_model.load_state_dict(checkpoint['model'])
    tqn_model.eval()
    return ked_encoder, text_encoder, tqn_model

def get_label_features(text_encoder, tokenizer, label_texts, device):
    text_batch = tokenizer(label_texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        label_features = text_encoder.encode_text(text_batch)  # shape: [num_labels, 768]
    return label_features

# ==== 3. 推理和保存 ====
def batch_predict_hardlabel_dataloader(ecg_loader, ecg_model, tqn_model, label_features, device="cuda"):
    hard_labels = []
    with torch.no_grad():
        for batch in tqdm(ecg_loader, desc='Inference'):
            batch = batch.float().to(device)  # [B, 12, 5000]
            feats = ecg_model(batch)          # [B, 768, seq_len]
            feats = feats.transpose(1,2)      # [B, seq_len, 768]
            out = tqn_model(feats, label_features)  # [B, num_labels, 2] or [B, num_labels]
            if out.shape[-1] == 2:
                soft_prob = torch.softmax(out, dim=-1)[..., 1]  # [B, num_labels]
            else:
                soft_prob = torch.sigmoid(out)
            hard_label = (soft_prob > 0.5).long()  # 阈值化为0/1
            hard_labels.append(hard_label.cpu().numpy())
    hard_labels = np.concatenate(hard_labels, axis=0)
    return hard_labels

if __name__ == "__main__":
    # ========== 路径配置 ==========
    ckpt_path = "/data_C/sdb1/lyi/ked3/control-spiderman-ECGFM-KED-456810e/trained_model/checkpoints_finetune/finetune_sph.pt"
    bert_model_name = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/model/multimodal_encoder/emilyalsentzer/Bio_ClinicalBERT"
    label_csv = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/all_labels_normalized.csv"   # 包含标签文本
    label_column = "label"
    data_dir = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/edge_2/dataset/sph/split/"
    batch_size = 128
    tqn_layers = 7
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess_cfg = PreprocessCfg(seq_length=5000, duration=10, sampling_rate=500)
    test_transform = ecg_transform_v2(cfg=preprocess_cfg, is_train=False)

    # 加载云端模型和标签特征，只需一次
    ecg_model, text_encoder, tqn_model = load_cloud_model(
        ckpt_path, bert_model_name, device, tqn_layers=tqn_layers
    )
    df_label = pd.read_csv(label_csv)
    label_texts = df_label[label_column].tolist()
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    label_features = get_label_features(text_encoder, tokenizer, label_texts, device)

    import re
    def extract_num(filename):
        nums = re.findall(r'\d+', filename)
        return int(nums[-1]) if nums else -1

    file_list = [f for f in os.listdir(data_dir) if f.startswith("update_") and f.endswith(".csv")]
    file_list = sorted(file_list, key=extract_num)

    for csv_file in file_list:
        csv_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(csv_path)
        signal_paths = df['path'].tolist()
        dataset = ECGSignalDataset(signal_paths, transforms=test_transform)
        ecg_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        hard_labels = batch_predict_hardlabel_dataloader(
            ecg_loader, ecg_model, tqn_model, label_features, device
        )

        # 保存为csv格式，每行为 path,label  label为[0 0 1 ... 0 0]
        save_csv_path = os.path.join(data_dir, f"ecg_hardlabel_save_{csv_file.replace('.csv','.csv')}")
        with open(save_csv_path, 'w', newline='') as f:
            f.write('path,label\n')
            for path, label in zip(signal_paths, hard_labels):
                label_str = "[{}]".format(' '.join(str(int(x)) for x in label))
                f.write(f"{path},{label_str}\n")
        print(f"{csv_file} 推理完成，硬标签已保存为: {save_csv_path}")