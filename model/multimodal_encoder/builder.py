import os

import sys
# 这里换成 ked_encoder 实际所在的上级目录路径 
sys.path.append("/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/model/multimodal_encoder")
# 新增：导入你自定义的编码器
from ked_encoder import KEDEncoderWrapper
from xresnet1d_101 import xresnet1d101
import torch



def build_ecg_tower(pretrained_path=None, device='cuda'):
    ked_encoder = xresnet1d101(input_channels=12, kernel_size=5, use_ecgNet_Diagnosis='other')
    print("=== build_ecg_tower called ===")
    print("build_ecg_tower 里的 pretrained_path:", pretrained_path)
    if pretrained_path:
        print("===================================================================================")
        print("pretrained_path:", pretrained_path)
        checkpoint= torch.load(pretrained_path, map_location="cpu")
        missing, unexpected = ked_encoder.load_state_dict(checkpoint['ecg_model'],strict=False)
        import os
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            try:
                with open("ecg_weight_load_log.txt", "w") as f:
                    f.write(f"missing keys: {missing}\n")
                    f.write(f"unexpected keys: {unexpected}\n")
                print("Loaded ECG encoder weights.")
            except Exception as e:
                print("写文件失败：", e)
    ecg_encoder = KEDEncoderWrapper(ked_encoder)
    ecg_encoder = ecg_encoder.to(device)
    return ecg_encoder