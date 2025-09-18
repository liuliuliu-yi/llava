import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModel,BertConfig,AutoTokenizer
# from pytorch_pretrained_vit import ViT

from transformer_decoder import *

import numpy as np


class CLP_clinical(nn.Module):
    def __init__(self,
                 bert_model_name: str,
                 embed_dim: int = 768,
                 # freeze_layers: Union[Tuple[int, int], int] = None,
                 freeze_layers = [0,1]):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name, freeze_layers=freeze_layers)
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers=[0,1]):  # 12
        try:
            print(bert_model_name)
            config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)  # bert-base-uncased
            model = AutoModel.from_pretrained(bert_model_name, config=config)  # , return_dict=True)
            print("Text feature extractor:", bert_model_name)
            print("bert encoder layers:", len(model.encoder.layer))
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers:   # [0,1,2,3,4,5,6,7,8,9,10]
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
            for layer_idx in set(range(len(model.encoder.layer))) - set(freeze_layers):
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = True
        # if freeze_layers != -1:
        #     for layer_idx in range(
        #             len(model.encoder.layer) - freeze_layers):  # Freeze all layers except last freeze_layers layers
        #         for param in list(model.encoder.layer[layer_idx].parameters()):
        #             param.requires_grad = False
        return model

    def encode_text(self, text):
        # input batch_size,token, return batch_size,dim
        output = self.bert_model(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
        last_hidden_state, pooler_output, hidden_states = output[0], output[1], output[2]
        encode_out = self.mlp_embed(pooler_output)
        return encode_out

    def encode_origin_text(self, text):
        with torch.no_grad():
            output = self.bert_model(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
        return output[0]

    # 代码中好像没用forward函数，而是都用的encode_text来获取编码，因此可学习参数还有三层mlp
    def forward(self, text1, text2):
        text1_features = self.encode_text(text1)
        text2_features = self.encode_text(text2)
        text1_features = F.normalize(text1_features, dim=-1)
        text2_features = F.normalize(text2_features, dim=-1)
        return text1_features, text2_features, self.logit_scale.exp()
    
if __name__ == "__main__":
    # 1. 实例化CLP_clinical，指定BERT模型名称
    bert_model_name = "/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/model/multimodal_encoder/emilyalsentzer/Bio_ClinicalBERT" 
    text_encoder = CLP_clinical(bert_model_name=bert_model_name, embed_dim=768)

    # 加载权重
    # checkpoint = torch.load("best_model.pth", map_location='cpu')
    # text_encoder.load_state_dict(checkpoint)
    # text_encoder.eval()

    # 2. 构造标签文本列表  全部标签集
    #label_texts = ["Normal ECG", "Myocardial Infarction", "Atrial Fibrillation"]

    # 读取 CSV 文件，假设文件和代码在同一目录，若不在需指定完整路径
    df = pd.read_csv('/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/model/multimodal_encoder/all_label_standard.csv')  
    # 假设要提取名为“列名”的列数据转为列表，将“列名”替换成实际的列标题
    column_name = "label"  
    label_texts = df[column_name].tolist()  

    # 3. 使用tokenizer对文本批量编码
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    text_batch = tokenizer(label_texts, padding=True, truncation=True, return_tensors='pt')

    # 4. 获取标签特征（embedding），输出shape为 [标签数, 768]
    label_features = text_encoder.encode_text(text_batch)
    print(label_features.shape)  # torch.Size([3, 768])