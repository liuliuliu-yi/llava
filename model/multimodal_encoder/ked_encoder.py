import torch
import torch.nn as nn
from xresnet1d_101 import xresnet1d101
class KEDEncoderWrapper(nn.Module):
    def __init__(self, ked_encoder, token_num=100):
        super().__init__()
        self.ked_encoder = ked_encoder
        self.adapt_pool = nn.AdaptiveAvgPool1d(token_num)
       

        # 记录模型关键配置参数
        self.hidden_size = 768   # 按你的模型实际输出设定
        self.token_num = token_num


        # 参数全部冻结（不参与训练）
        for p in self.ked_encoder.parameters():
            p.requires_grad = False
    

    @torch.no_grad()  # 前向推理不计算梯度
    def forward(self, x):
        feats = self.ked_encoder(x)           # [batch, 768, 157]
        token_emb = self.adapt_pool(feats)    # [batch, 768, 100]
        token_emb = token_emb.transpose(1,2)  # [batch, 100, 768]
        global_emb = token_emb.mean(dim=1)    # [batch, 768]
        return global_emb, token_emb

if __name__ == '__main__':
    ked_encoder = xresnet1d101(input_channels=12, kernel_size=5, use_ecgNet_Diagnosis='other')
   
    # 包装成兼容ECG-Chat格式
    ecg_encoder = KEDEncoderWrapper(ked_encoder)
    # forward推理
    x = torch.randn(8, 12, 5000)  # 假设batch=8, 12导联, 5000采样点
    global_emb, token_emb = ecg_encoder(x)
    #global_emb=ecg_encoder(x)
    print(global_emb.shape)  # torch.Size([8, 768])
    print(token_emb.shape)   # torch.Size([8, 100, 768])

