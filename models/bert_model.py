import sys
import torch
import os
from torch import nn
import torch.nn.functional as F
from TorchCRF import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50, resnet101, resnet152
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import numpy as np

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)

    def forward(self, x, aux_imgs=None):
        prompt_guids = self.get_resnet_prompt(x)
        if aux_imgs is not None:
            aux_prompt_guids = []
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])
            for i in range(len(aux_imgs)):
                aux_prompt_guid = self.get_resnet_prompt(aux_imgs[i])
                aux_prompt_guids.append(aux_prompt_guid)
            return prompt_guids, aux_prompt_guids
        return prompt_guids, None

    def get_resnet_prompt(self, x):
        prompt_guids = []
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)
            if 'layer' in name:
                bsz, channel, ft, _ = x.size()
                kernel = ft // 2
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)
                prompt_guids.append(prompt_kv)
        return prompt_guids

config = AutoConfig.from_pretrained("roberta-large", num_labels = 23)

class I2SRMModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(I2SRMModel, self).__init__()
        self.roberta = AutoModel.from_pretrained("roberta-large", config=config)
        self.roberta.resize_token_embeddings(
            len(tokenizer))
        self.args = args
        self.dropout = nn.Dropout(0.5)

        self.linear_kl = nn.Linear(1000, self.roberta.config.hidden_size * 2)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.kldlambda = 0.5

        self.classifier = nn.Linear(self.roberta.config.hidden_size * 2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<*>")
        self.head_end = tokenizer.convert_tokens_to_ids("</*>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<#>")
        self.tail_end = tokenizer.convert_tokens_to_ids("</#>")
        self.tokenizer = tokenizer
        if self.args.use_prompt:
            self.image_model = ImageModel()
            self.encoder_conv = nn.Sequential(
                nn.Linear(in_features=3840, out_features=800),
                nn.Tanh(),
                nn.Linear(in_features=800, out_features=4 * 2 * 1024)
            )
            self.gates = nn.ModuleList([nn.Linear(4 * 1024 * 2, 4) for i in range(24)])
        self.encoder_layers = torch.nn.TransformerEncoderLayer(2048, 4, 2048, 0.2)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            images=None,
            aux_imgs=None,
            mode=None
    ):
        bsz = input_ids.size(0)
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_guids = None
            prompt_attention_mask = attention_mask

        output = self.roberta(
            input_ids=input_ids,
            attention_mask=prompt_attention_mask,
            past_key_values=prompt_guids,
            output_attentions=True,
            return_dict=True
        )
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2 * hidden_size)
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            if self.head_end in input_ids[i]:
                head_end_idx = input_ids[i].eq(self.head_end).nonzero().item()
            else:
                head_end_idx = input_ids[i].eq(self.head_start).nonzero().item() + 2
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            if self.tail_end in input_ids[i]:
                tail_end_idx = input_ids[i].eq(self.tail_end).nonzero().item()
            else:
                tail_end_idx = input_ids[i].eq(
                    self.tail_start).nonzero().item() + 2
            head_hidden = last_hidden_state[i, head_idx + 1:head_end_idx, :].mean(
                dim=0).squeeze()
            tail_hidden = last_hidden_state[i, tail_idx + 1:tail_end_idx, :].mean(
                dim=0).squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)

        img_reg = self.image_model.resnet_output(images)
        img_reg = self.linear_kl(img_reg)
        modal1_pic = F.log_softmax(img_reg)
        modal2_lan = F.softmax(entity_hidden_state)
        kl_loss = self.kl_loss(modal1_pic, modal2_lan)

        entity_hidden_state_bf, labels_bf = self.intersamples(entity_hidden_state, labels, mode)
        entity_hidden_state_mixup, labels_a, labels_b, lam = self.mixup_data_original(entity_hidden_state_bf, labels_bf, mode=mode)
        # Truncate
        entity_hidden_state_bf = entity_hidden_state_bf[:int((0.5 + 0.6*0.5) * entity_hidden_state_bf.size(0)), :]
        labels_bf =labels_bf[:int((0.5 + 0.6*0.5) * entity_hidden_state_bf.size(0))]

        logits = self.classifier(entity_hidden_state_mixup)
        logits_bf = self.classifier(entity_hidden_state_bf)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss_bf = loss_fn(logits_bf, labels_bf.view(-1))
            loss = self.mixup_criterion(logits, labels_a, labels_b, lam)
            logit_all = torch.cat((logits, logits_bf), dim=0)
            return loss_bf + loss + self.kldlambda * kl_loss, logits
        return logits

    def intersamples(self, entity_hidden_state, labels, mode):
        old_feat = entity_hidden_state
        if mode == 'train':
            entity_hidden_state = entity_hidden_state.unsqueeze(1)
            entity_hidden_state = self.encoder_layers(entity_hidden_state)
            entity_hidden_state = entity_hidden_state.squeeze(1)
            entity_hidden_state = torch.cat([old_feat, entity_hidden_state], dim=0)
            labels = torch.cat([labels, labels], dim=0)
        return entity_hidden_state, labels

    def mixup_data_original(self, x, y, mode, alpha=1.0, delta=0.7,use_cuda=True):
        x, y = self.sample_combination(x, y, delta=delta)
        if mode == 'train':
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1
            batch_size = x.size()[0]
            if use_cuda:
                index = torch.randperm(batch_size).cuda()
            else:
                index = torch.randperm(batch_size)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            # Truncate
            mixed_x = mixed_x[:int(0.4 * mixed_x.size(0)), :]
            y_a = y_a[:int(0.4 * y_a.size(0))]
            y_b = y_b[:int(0.4 * y_b.size(0))]
            return mixed_x, y_a, y_b, lam
        return x, y, y, 1

    def sample_combination(self, x, y, delta=0.7):
        num_rows = x.size(0)
        x_1 = x[:int(delta / 2 * num_rows), :]
        y_1 = y[:int(delta / 2 * num_rows)]
        x_2 = x[int((1 - delta / 2) * num_rows):num_rows, :]
        y_2 = y[int((1 - delta / 2) * num_rows):num_rows]
        a = torch.cat([x_1, x_2], dim=0)
        b = torch.cat([y_1, y_2], dim=0)
        return a, b

    def mixup_criterion(self, pred, y_a, y_b, lam):
        criterion = nn.CrossEntropyLoss()
        return lam * criterion(pred, y_a.view(-1)) + (1 - lam) * criterion(pred, y_b.view(-1))

    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in
                            aux_prompt_guids]
        prompt_guids = self.encoder_conv(prompt_guids)
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in
                            aux_prompt_guids]
        split_prompt_guids = prompt_guids.split(1024 * 2, dim=-1)
        split_aux_prompt_guids = [aux_prompt_guid.split(1024 * 2, dim=-1) for aux_prompt_guid in
                                  aux_prompt_guids]
        sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4
        result = []
        for idx in range(24):
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)
            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])
            aux_key_vals = []
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1),
                                                             split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(1024, dim=-1)
            key, value = key_val[0].reshape(bsz, 16, -1, 64).contiguous(), key_val[1].reshape(bsz, 16, -1,
                                                                                              64).contiguous()
            temp_dict = (key, value)
            result.append(temp_dict)
        return result