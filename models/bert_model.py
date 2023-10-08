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

class I2SRMREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(I2SRMREModel, self).__init__()
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

class I2SRMNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(I2SRMNERModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.roberta = AutoModel.from_pretrained("roberta-large", config=config)
        self.roberta_config = self.roberta.config
        if args.use_prompt:
            self.image_model = ImageModel()
            self.encoder_conv =  nn.Sequential(
                            nn.Linear(in_features=3840, out_features=800),
                            nn.Tanh(),
                            nn.Linear(in_features=800, out_features=4*2*1024)
                            )
            self.gates = nn.ModuleList([nn.Linear(4*1024*2, 4) for i in range(24)])

        self.num_labels  = len(label_list)  # pad
        # print('self.num_labels是', self.num_labels) # 13
        self.crf = CRF(self.num_labels) # self.crf = CRF(self.num_labels, batch_first=True)
        # self.fc = nn.Linear(self.roberta.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)

        batchformerhidden = 150
        # self.encoder_layers = torch.nn.TransformerEncoderLayer(batchformerhidden * 80, 4, batchformerhidden * 80, 0.3)  # twitter15的长度是80。 dropout原来默认0.1, cvpr里是0.5。
        self.encoder_layers = torch.nn.TransformerEncoderLayer(batchformerhidden * 128, 2, batchformerhidden * 128, 0.1)  # twitter17的长度是128。dropout原来默认0.1, cvpr里是0.5。
        print('使用batchformer, head=4, dropout=0.3。batchformer中隐藏层维度是', batchformerhidden)
        self.fc = nn.Linear(self.roberta.config.hidden_size, batchformerhidden)
        self.linear = nn.Linear(batchformerhidden, self.num_labels)

        # KLD 损失
        self.linear_kl = nn.Linear(1000, self.roberta.config.hidden_size)  # 为图片特征加入这个线性层。
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.kldlambda = 1.0
        print('KLD损失前面的系数是', self.kldlambda)

    def forward(self, input_ids=None, attention_mask=None, labels=None, images=None, aux_imgs=None, mode=None):
        # print('input_ids: ', input_ids.size(), input_ids) # input_ids:  torch.Size([8, 80]) tensor([[  101,  2193,  2028,  ..., 2575, 102,     0,
        # print('labels: ', labels.size(), labels) # labels:  torch.Size([8, 80]) tensor([[11,  1,  1,  1,  1,  8,  1,  1, 10, 10, 10,  1,  1,  1,  1,  1, 10, 10, ..., 10, 10, 12,  0,  0,  0,
        # print('mode: ', mode) # mode:  "train"

        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            # attention_mask: bsz, seq_len
            # prompt attention， attention mask
            bsz = attention_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None

        bert_output = self.roberta(input_ids=input_ids,
                            attention_mask=prompt_attention_mask,
                            past_key_values=prompt_guids,
                            return_dict=True)
        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden # sequence_output:  torch.Size([8, 80, 1024])

        # kl 正则。
        img_reg = self.image_model.resnet_output(images)  # resnet50的输出： img_reg:  torch.Size([8, 1000])
        img_reg = self.linear_kl(img_reg)
        modal1_pic = F.log_softmax(img_reg)
        modal2_lan = F.softmax(sequence_output[:, 0, :])
        kl_loss = self.kl_loss(modal1_pic, modal2_lan) # kl_loss:  tensor(0.6235, device='cuda:0', grad_fn=<DivBackward0>)

        # 20220831 使用过batchformer.
        emissions = self.fc(sequence_output)  # bsz, len, labels
        # print('sequence_output: ', sequence_output.size(), 'emissions: ', emissions.size()) # sequence_output:  torch.Size([8, 80, 1024]) emissions:  torch.Size([8, 80, 150])

        # if mode=="train":
        # print('emissions in mode train: ', emissions.size()) # twitter15的长度是80. emissions in mode train:  torch.Size([8, 80, 150])。twitter17的长度是128emissions:  torch.Size([8, 128, 150])
        emissions, labels, attention_mask = self.batchformer(emissions, labels, attention_mask, mode=mode)

        # 增加ICLR2018 mixup方法。
        emissions, labels, attention_mask = self.mixup_data(emissions, labels, attention_mask, mode=mode)

        emissions = self.linear(emissions)
        # print('emissions: ', emissions.size()) # emissions:  torch.Size([32, 80, 13])

        logits = self.crf.viterbi_decode(emissions, attention_mask.byte()) # logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * torch.mean(self.crf(emissions, labels, mask=attention_mask.byte()))  # loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            # print('loss是', type(loss), loss.size(), loss) # loss是 <class 'torch.Tensor'> torch.Size([8]) tensor([107.7501, 115.2792,  92.3825,  88.1822,  88.6666, 125.3268,  59.3856, 143.7558], device='cuda:0', grad_fn=<MulBackward0>)
            loss += self.kldlambda * kl_loss # 加上KLD。
            # sys.exit(0)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    def batchformer(self, sequence_output, labels, attention_mask, mode):  # mode in "train" "dev"。改变输入和标签。
        # sequence_output:  torch.Size([8, 80, 768]), batchsize=8.
        # labels:  torch.Size([8, 80])
        old_feat = sequence_output
        # print('mode:', mode) # mode: train
        if mode == 'train':
            entity_hidden_state = sequence_output.view(sequence_output.size()[0], -1).unsqueeze(1)
            # print('entity_hidden_state: ', entity_hidden_state.size()) # twitter17是entity_hidden_state:  torch.Size([8, 1, 19200]) 128*150
            entity_hidden_state = self.encoder_layers(entity_hidden_state)
            entity_hidden_state = entity_hidden_state.squeeze(1).view(old_feat.size()[0], old_feat.size()[1], -1)
            sequence_output = torch.cat([old_feat, entity_hidden_state], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            # print('sequence_output用batchformer: ', sequence_output.size()) # sequence_output用batchformer:  torch.Size([16, 80, 200]), batchsize是原来的2倍。
            # print('labels用batchformer: ', labels.size()) # labels用batchformer: torch.Size([16, 80])
        return sequence_output, labels, attention_mask

    def mixup_data(self, x, y, attention_mask, mode, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        # print('x: ', x.size()) # x:  torch.Size([16, 80, 150])
        # print('y: ', y.size(), y) # y:  torch.Size([16, 80]) tensor([[11,  1,  1,  ...,  0,  0,  0], ..., [11,  8,  2,  ...,  0,  0,  0], [11,  8, 10,  ...,  0,  0,  0]], device='cuda:0')。
        # print('attention_mask1: ', attention_mask.size(), attention_mask) # twitter15数据集是  attention_mask1:  torch.Size([16, 80]) tensor([[1, 1, 1,  ..., 0, 0, 0],
        # twitter17数据集是 attention_mask1:  torch.Size([16, 128]) tensor([[1, 1, 1,  ..., 0, 0, 0], ..., [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0')

        if mode == 'train':
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            # print('attention_mask2: ', attention_mask.size(), attention_mask) # attention_mask2:  torch.Size([32, 80]) tensor([[1, 1, 1,  ..., 0, 0, 0],
            # sys.exit(0)
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

            # 重新组合
            origx_and_mixedx = torch.cat((x, mixed_x), dim=0)
            mixed_y = lam * y + (1 - lam) * y[index, :]
            origy_and_mixedroundy = torch.cat((y, torch.round(mixed_y)), dim=0).long() # 注意这里 torch.round(torch.tensor(0.50))是0，而torch.round(torch.tensor(1.50))是2。

            # print('origx_and_mixedx: ', origx_and_mixedx.size()) # origx_and_mixedx:  torch.Size([32, 80, 150])
            # print('origy_and_mixedroundy: ', origy_and_mixedroundy.size(), y) # origy_and_mixedroundy:  torch.Size([32, 80]) tensor([[11,  1,  1,  ...,  0,  0,  0], ..., [11,  8, 10,  ...,  0,  0,  0]], device='cuda:0')
            # for i in range(len(origy_and_mixedroundy)):
            #     print(i, origy_and_mixedroundy[i])
            # sys.exit(0)
            return origx_and_mixedx, origy_and_mixedroundy, attention_mask
        return x, y, attention_mask

    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....

        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]

        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        split_prompt_guids = prompt_guids.split(1024*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(1024*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]

        result = []
        for idx in range(24):  # 12
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(1024, dim=-1)
            key, value = key_val[0].reshape(bsz, 16, -1, 64).contiguous(), key_val[1].reshape(bsz, 16, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result
