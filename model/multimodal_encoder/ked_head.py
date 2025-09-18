import torch
from torch import nn

# from pytorch_pretrained_vit import ViT

from transformer_decoder import *
from visualization import visualization_tsne

import numpy as np



class TQNModel(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 class_num: int = 2,
                 num_layers: int = 3
                 ):
        super().__init__()
        self.d_model = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        decoder_layer = TransformerDecoderLayer(self.d_model, 4, 1024,
                                                0.1, 'relu', normalize_before=True)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_layers, self.decoder_norm,
                                          return_intermediate=False)
        self.dropout_feas = nn.Dropout(0.1)

        self.mlp_head = nn.Sequential(  # nn.LayerNorm(768),
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, class_num)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, ecg_features, text_features, return_atten=False): #（64，250，768） （5，768）
        # image_features (batch_size,patch_num,dim)
        # text_features (query_num,dim)
        batch_size = ecg_features.shape[0]
        ecg_features = ecg_features.transpose(0, 1)     # (250, 64, 768)
        text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1)     # (5,64,768)
        ecg_features = self.decoder_norm(ecg_features)
        text_features = self.decoder_norm(text_features)
        features, atten_map = self.decoder(text_features, ecg_features,
                                memory_key_padding_mask=None, pos=None, query_pos=None)     # (40, 32, 768)
        features = self.dropout_feas(features).transpose(0, 1)  # (32,5,768)
        out = self.mlp_head(features)  # (32,5,2)  做二分类，判断是还是不是这个类别
        if return_atten:
            return out, atten_map
        else:
            return out
    def visual_tsne(self, ecg_features, text_features, label_list):
        batch_size = ecg_features.shape[0]
        ecg_features = ecg_features.transpose(0, 1)  # (250, 64, 768)
        text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1)  # (5,64,768)
        ecg_features = self.decoder_norm(ecg_features)
        text_features = self.decoder_norm(text_features)
        features = self.decoder(text_features, ecg_features,
                                memory_key_padding_mask=None, pos=None, query_pos=None)  # (40, 32, 768)
        features = self.dropout_feas(features).transpose(0, 1)  # (32,5,768)
        # features = features.max(1) # (32,768)
        # features = torch.nn.functional.max_pool1d(features, kernel_size=features.shape[1])

        m = nn.MaxPool1d(features.shape[1])
        features = m(features.transpose(1,2)).squeeze()
        visualization_tsne(features,label_list)





if __name__ == "__main__":
    model = TQNModel(
    embed_dim=768,   # 特征维度（和ecg特征输出、标签特征一致）
    class_num=2,     # 每个标签二分类，可设为1（sigmoid）或2（softmax）
    num_layers=7     # decoder的层数
)
    #ked_encoder的输出 [batch, 768, 157] 需交换位置
    ecg_features = torch.randn(8, 157, 768)       # batch=8, seq_len=157, embed_dim=768

    # 标签特征由文本编码器（如BERT）编码，假设有105个标签
    label_features = torch.randn(105, 768)          # num_labels=5, embed_dim=768

    # 3. 前向推理
    out = model(ecg_features, label_features)
    soft_label = F.softmax(out, dim=-1)  
    print(soft_label.shape)  # [8, 105, 2]，对应 batch=8, num_labels=105, 二分类