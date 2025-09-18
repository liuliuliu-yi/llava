#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_ecg_tower
from .multimodal_projector.builder import build_ecg_projector

from llava.constants import IGNORE_INDEX, ECG_TOKEN_INDEX, DEFAULT_ECG_PATCH_TOKEN, DEFAULT_ECG_START_TOKEN, DEFAULT_ECG_END_TOKEN,DEFAULT_ECG_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        self.ecg_tower = None
        self.mm_projector = None


    def get_ecg_tower(self):
        ecg_tower = getattr(self, 'ecg_tower', None)
        if isinstance(ecg_tower, list):
            ecg_tower = ecg_tower[0]
        return ecg_tower

    def initialize_ecg_modules(self, model_args, fsdp=None):
        """
        初始化ECG相关模块（编码器与投影器），支持分布式等场景。
        model_args: 通常与config类似，包含ecg_tower、mm_projector_type等参数。
        fsdp: 分布式训练相关参数，默认None。
        """
        
        # 1. 配置参数同步到config
        self.config.mm_ecg_tower = getattr(model_args, "ecg_tower", None)
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "mlp2x_gelu")
        self.config.mm_hidden_size = 768
        self.config.mm_patch_merge_type = getattr(model_args, "mm_patch_merge_type", None)
        
        # 2. ECG编码器初始化及加载
        ecg_pretrained_path = getattr(model_args, "ecg_pretrained_path", None)
        print("传入 build_ecg_tower 的 pretrained_path:", ecg_pretrained_path)
        if getattr(self, "ecg_tower", None) is None:
            ecg_tower = build_ecg_tower(
                pretrained_path=ecg_pretrained_path,
                device=getattr(model_args, "device", "cuda")
            )
            if fsdp and hasattr(fsdp, "__len__") and len(fsdp) > 0:
                self.ecg_tower = [ecg_tower]
            else:
                self.ecg_tower = ecg_tower
        else:
            if fsdp and hasattr(fsdp, "__len__") and len(fsdp) > 0:
                ecg_tower = self.ecg_tower[0]
            else:
                ecg_tower = self.ecg_tower
            

          # 3. ECG投影层初始化
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_ecg_projector(self.config)
            
        else:
            # 若已存在projector且被冻结，解冻参数（可选）
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # 4. 加载预训练的projector权重（可选）
        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}
            # 这里的'mm_projector'需与保存权重时的key一致
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))


    def encode_ecg(self, ecg_wave):
        """
        输入: ecg_wave, shape [B, 12, 5000]（或你的实际输入shape）
        输出: 映射到LLM hidden_size的token特征 [B, token_num, LLM_hidden_size]
        """
        ecg_tower = self.get_ecg_tower()
        if ecg_tower is None:
            raise ValueError("ECG tower not initialized!")
        global_emb, token_emb = ecg_tower(ecg_wave)   # token_emb: [B, token_num, ecg_hidden_size]
        projected_token_emb = self.mm_projector(token_emb)  # [B, token_num, LLM_hidden_size]
        return projected_token_emb

# def unpad_image(tensor, original_size):
#     """
#     Unpads a PyTorch tensor of a padded and resized image.
#
#     Args:
#     tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
#     original_size (tuple): The original size of PIL image (width, height).
#
#     Returns:
#     torch.Tensor: The unpadded image tensor.
#     """
#     original_width, original_height = original_size
#     current_height, current_width = tensor.shape[1:]
#
#     original_aspect_ratio = original_width / original_height
#     current_aspect_ratio = current_width / current_height
#
#     if original_aspect_ratio > current_aspect_ratio:
#         scale_factor = current_width / original_width
#         new_height = int(original_height * scale_factor)
#         padding = (current_height - new_height) // 2
#         unpadded_tensor = tensor[:, padding:current_height - padding, :]
#     else:
#         scale_factor = current_height / original_height
#         new_width = int(original_width * scale_factor)
#         padding = (current_width - new_width) // 2
#         unpadded_tensor = tensor[:, :, padding:current_width - padding]
#
#     return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_ecg_tower(self):
        return self.get_model().get_ecg_tower()

    def encode_ecg(self, ecg_wave):
        ecg_tower = self.get_model().get_ecg_tower()
        global_emb, token_emb = ecg_tower(ecg_wave)
        projected_token_emb = self.get_model().mm_projector(token_emb)
        return projected_token_emb

    def prepare_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, past_key_values, labels,
    ecg_wave  # shape: [B, 12, 5000] 或你的实际输入
):
        """
        多模态输入拼接：将ECG特征按ECG_TOKEN_INDEX插入文本embedding序列。
        """

        print("[DEBUG] prepare_inputs_labels_for_multimodal called")
        print("input_ids shape:", input_ids.shape if input_ids is not None else None)
        print("ecg_wave shape:", ecg_wave.shape if ecg_wave is not None else None)

        
        ecg_tower = self.get_ecg_tower()
        if ecg_tower is None or ecg_wave is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # 1. 得到ECG特征
        ecg_features = self.encode_ecg(ecg_wave)  # [B, token_num, LLM_hidden_size]

        # 2. 保证mask/positions/labels不为None
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)   
        # 3. 按mask去除padding
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # 4. 按ECG_TOKEN_INDEX插入ECG特征
        new_input_embeds = []
        new_labels = []
        cur_ecg_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_ecgs = (cur_input_ids == ECG_TOKEN_INDEX).sum()
            if num_ecgs == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue
            
            # 找到每个ECG token下标
            ecg_token_indices = [-1] + torch.where(cur_input_ids == ECG_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noecg = []
            cur_labels = labels[batch_idx]
            cur_labels_noecg = []
            for i in range(len(ecg_token_indices) - 1):
                cur_input_ids_noecg.append(cur_input_ids[ecg_token_indices[i]+1:ecg_token_indices[i+1]])
                cur_labels_noecg.append(cur_labels[ecg_token_indices[i]+1:ecg_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noecg]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noecg))
            cur_input_embeds_no_ecg = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_ecgs + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_ecg[i])
                cur_new_labels.append(cur_labels_noecg[i])
                if i < num_ecgs:
                    # 插入ECG特征
                    cur_ecg_features = ecg_features[cur_ecg_idx]
                    cur_ecg_idx += 1
                    cur_new_input_embeds.append(cur_ecg_features)
                    cur_new_labels.append(torch.full((cur_ecg_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # 5. 截断和padding
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_ecg_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_ecg_patch_token:
            tokenizer.add_tokens([DEFAULT_ECG_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_ecg_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_ECG_START_TOKEN, DEFAULT_ECG_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_ecg_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
