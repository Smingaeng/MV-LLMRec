import torch as t
from torch import nn
import numpy as np
import scipy.sparse as sp
import torch_sparse
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, ssl_con_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
import torch.nn.functional as F
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class LightGCN_mv(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_mv, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        self.intent_num = configs['model']['intent_num']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.user_intent = t.nn.Parameter(init(t.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = t.nn.Parameter(init(t.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        
        # --- Conformity ---
        self.conformity_num = configs['model']['conformity_num'] 
        self.user_conformity = t.nn.Parameter(init(t.empty(self.embedding_size, self.conformity_num)), requires_grad=True)
        self.item_conformity = t.nn.Parameter(init(t.empty(self.embedding_size, self.conformity_num)), requires_grad=True)
        
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # prepare for adaptive mask
        rows = data_handler.trn_mat.tocoo().row
        cols = data_handler.trn_mat.tocoo().col
        new_rows = np.concatenate([rows, cols + self.user_num], axis=0)
        new_cols = np.concatenate([cols + self.user_num, rows], axis=0)
        plain_adj = sp.coo_matrix(
            (np.ones(len(new_rows)), (new_rows, new_cols)),
            shape=[self.user_num + self.item_num, self.user_num + self.item_num]
        ).tocsr().tocoo()
        self.all_h_list = t.LongTensor(list(plain_adj.row)).cuda()
        self.all_t_list = t.LongTensor(list(plain_adj.col)).cuda()
        self.A_in_shape = plain_adj.shape

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.kd_int_weight = self.hyper_config['kd_int_weight']
        self.kd_int_temperature = self.hyper_config['kd_int_temperature']
        self.kd_int_weight_2 = self.hyper_config['kd_int_weight_2']
        self.kd_int_weight_3 = self.hyper_config['kd_int_weight_3']
        # --- Conformity ---
        self.kd_conf_weight = self.hyper_config['kd_conf_weight']
        self.kd_conf_temperature = self.hyper_config['kd_conf_temperature']
        self.kd_conf_weight_3 = self.hyper_config.get('kd_conf_weight_3', 0.0) # conf momentum loss weight, 없으면 0으로
        # --- HCL ---
        self.hcl_weight = self.hyper_config.get('hcl_weight', 0.001)
        self.hcl_temperature = self.hyper_config.get('hcl_temperature', 0.2)
        
        # semantic-embeddings (Profile)
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        # intent information
        self.usrint_embeds = t.tensor(configs['usrint_embeds']).float().cuda()
        self.itmint_embeds = t.tensor(configs['itmint_embeds']).float().cuda()
        self.int_mlp = nn.Sequential(
            nn.Linear(self.usrint_embeds.shape[1], (self.usrint_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrint_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        self.int_mlp_m = nn.Sequential(
            nn.Linear(self.usrint_embeds.shape[1], (self.usrint_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrint_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        
        # Conformity Embeddings 및 MLP
        self.usrconf_embeds = t.tensor(configs['usrconf_embeds']).float().cuda()
        self.itmconf_embeds = t.tensor(configs['itmconf_embeds']).float().cuda()
        self.conf_mlp = nn.Sequential(
            nn.Linear(self.usrconf_embeds.shape[1], (self.usrconf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrconf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        # --- Conformity Momentum MLP ---
        self.conf_mlp_m = nn.Sequential(
            nn.Linear(self.usrconf_embeds.shape[1], (self.usrconf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrconf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        self.gate_network = nn.Sequential(
            nn.Linear(self.embedding_size * 3, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, 3),
            nn.Softmax(dim=1)
        )

        self.model_pairs = [
            [self.int_mlp, self.int_mlp_m],
            [self.conf_mlp, self.conf_mlp_m]
        ]
        self.momentum = 0.999
        self._init_weight()
        self.copy_params()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.int_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.int_mlp_m:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.conf_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.conf_mlp_m:
            if isinstance(m, nn.Linear):
                init(m.weight)
                
    @t.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @t.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = t.nn.functional.normalize(head_embeddings)
        tail_embeddings = t.nn.functional.normalize(tail_embeddings)
        edge_alpha = (t.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(
            row=self.all_h_list,
            col=self.all_t_list,
            value=edge_alpha,
            sparse_sizes=self.A_in_shape
        ).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_indices = t.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha
        return G_indices, G_values

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None

        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)

        iaa_embeds_list = []
        caa_embeds_list = []

        for i in range(self.layer_num):
            # 1. Standard GCN propagation
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

            current_layer_embeds = embeds_list[i]
            u_embeds, i_embeds = t.split(current_layer_embeds, [self.user_num, self.item_num], 0)
            
            # 2. Intent-aware Information Aggregation (IAA)
            u_int_embeds = t.softmax(u_embeds @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeds = t.softmax(i_embeds @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeds = t.concat([u_int_embeds, i_int_embeds], dim=0)
            
            head_embeds_int = t.index_select(int_layer_embeds, 0, self.all_h_list)
            tail_embeds_int = t.index_select(int_layer_embeds, 0, self.all_t_list)
            intent_indices, intent_values = self._adaptive_mask(head_embeds_int, tail_embeds_int)
            iaa_layer_embeds = torch_sparse.spmm(intent_indices, intent_values, self.A_in_shape[0], self.A_in_shape[1], current_layer_embeds)
            iaa_embeds_list.append(iaa_layer_embeds)

            # 3. Conformity-aware Information Aggregation (CAA)
            u_conf_embeds = t.softmax(u_embeds @ self.user_conformity, dim=1) @ self.user_conformity.T
            i_conf_embeds = t.softmax(i_embeds @ self.item_conformity, dim=1) @ self.item_conformity.T
            conf_layer_embeds = t.concat([u_conf_embeds, i_conf_embeds], dim=0)

            head_embeds_conf = t.index_select(conf_layer_embeds, 0, self.all_h_list)
            tail_embeds_conf = t.index_select(conf_layer_embeds, 0, self.all_t_list)
            conformity_indices, conformity_values = self._adaptive_mask(head_embeds_conf, tail_embeds_conf)
            caa_layer_embeds = torch_sparse.spmm(conformity_indices, conformity_values, self.A_in_shape[0], self.A_in_shape[1], current_layer_embeds)
            caa_embeds_list.append(caa_layer_embeds)

        gcn_embeds = sum(embeds_list)
        iaa_embeds = sum(iaa_embeds_list)
        caa_embeds = sum(caa_embeds_list)
        
        # --- Gating Adaptive Fusion ---
        stacked_embeds = t.stack([gcn_embeds, iaa_embeds, caa_embeds], dim=1)
        flat_embeds = stacked_embeds.view(-1, self.embedding_size * 3)
        gate_weights = self.gate_network(flat_embeds)
        gate_weights = gate_weights.unsqueeze(-1)
        final_embeds = t.sum(stacked_embeds * gate_weights, dim=1)

        self.final_embeds = final_embeds

        disentangled_embeds = {
            'iaa': iaa_embeds,
            'caa': caa_embeds,
            'gcn': gcn_embeds 
        }
        return final_embeds[:self.user_num], final_embeds[self.user_num:], disentangled_embeds

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        ancs, poss, negs = batch_data
        user_embeds, item_embeds, disentangled_embeds = self.forward(self.adj, self.keep_rate)
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)

        # BPR Loss & Regularization Loss
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)

        # KD Loss (Profile)
        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)
        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, itmprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds, negprf_embeds, itmprf_embeds, self.kd_temperature)
        kd_loss = self.kd_weight * (kd_loss / anc_embeds.shape[0])

        # KD Loss (Intent)
        iaa_embeds = disentangled_embeds['iaa']
        user_iaa_embeds, item_iaa_embeds = t.split(iaa_embeds, [self.user_num, self.item_num], 0)
        anc_iaa_embeds, pos_iaa_embeds, neg_iaa_embeds = self._pick_embeds(user_iaa_embeds, item_iaa_embeds, batch_data)
        usrint_embeds = self.int_mlp(self.usrint_embeds)
        itmint_embeds = self.int_mlp(self.itmint_embeds)
        ancint_embeds, posint_embeds, negint_embeds = self._pick_embeds(usrint_embeds, itmint_embeds, batch_data)
        kd_int_loss = cal_infonce_loss(anc_iaa_embeds, ancint_embeds, usrint_embeds, self.kd_int_temperature) + \
                      cal_infonce_loss(pos_iaa_embeds, posint_embeds, itmint_embeds, self.kd_int_temperature) + \
                      cal_infonce_loss(neg_iaa_embeds, negint_embeds, itmint_embeds, self.kd_int_temperature)
        kd_int_loss = self.kd_int_weight * (kd_int_loss / anc_embeds.shape[0])

        # KD Loss (Conformity)
        caa_embeds = disentangled_embeds['caa']
        user_caa_embeds, item_caa_embeds = t.split(caa_embeds, [self.user_num, self.item_num], 0)
        anc_caa_embeds, pos_caa_embeds, neg_caa_embeds = self._pick_embeds(user_caa_embeds, item_caa_embeds, batch_data)
        usrconf_embeds_proj = self.conf_mlp(self.usrconf_embeds)
        itmconf_embeds_proj = self.conf_mlp(self.itmconf_embeds)
        ancconf_embeds, posconf_embeds, negconf_embeds = self._pick_embeds(usrconf_embeds_proj, itmconf_embeds_proj, batch_data)
        kd_conf_loss = cal_infonce_loss(anc_caa_embeds, ancconf_embeds, usrconf_embeds_proj, self.kd_conf_temperature) + \
                       cal_infonce_loss(pos_caa_embeds, posconf_embeds, itmconf_embeds_proj, self.kd_conf_temperature) + \
                       cal_infonce_loss(neg_caa_embeds, negconf_embeds, itmconf_embeds_proj, self.kd_conf_temperature)
        kd_conf_loss = self.kd_conf_weight * (kd_conf_loss / anc_embeds.shape[0])

        # ---  Hierarchical Contrastive Learning ---
        gcn_embeds = disentangled_embeds['gcn']
        user_gcn_embeds, _ = t.split(gcn_embeds, [self.user_num, self.item_num], 0)
        anc_gcn_embeds = user_gcn_embeds[ancs]

        hcl_loss_1 = cal_infonce_loss(anc_gcn_embeds, anc_iaa_embeds, user_iaa_embeds, self.hcl_temperature)
        hcl_loss_2 = cal_infonce_loss(anc_gcn_embeds, anc_caa_embeds, user_caa_embeds, self.hcl_temperature)
        hcl_loss_3 = cal_infonce_loss(anc_iaa_embeds, anc_caa_embeds, user_caa_embeds, self.hcl_temperature)
        
        hcl_loss = (hcl_loss_1 + hcl_loss_2 + hcl_loss_3) / anc_embeds.shape[0]
        hcl_loss *= self.hcl_weight

        # Contrastive Loss
        all_embeds = t.cat([user_embeds, item_embeds], dim=0)
        all_int_embeds = t.cat([usrint_embeds, itmint_embeds], dim=0)
        noise_1 = t.randn_like(all_int_embeds)
        noise_2 = t.randn_like(all_embeds)
        noise_embeds_1 = all_int_embeds + all_int_embeds * noise_1
        noise_embeds_2 = all_embeds + all_embeds * noise_2
        kd_int_2_loss =  ssl_con_loss(noise_embeds_1, noise_embeds_2)
        kd_int_2_loss *= self.kd_int_weight_2

        # Momentum Update & Loss
        self._momentum_update()
        
        # ITM Loss (Intent)
        usrint_embeds_m = self.int_mlp_m(self.usrint_embeds)
        itmint_embeds_m = self.int_mlp_m(self.itmint_embeds)
        int_embeds_m = t.cat([usrint_embeds_m, itmint_embeds_m], dim=0)
        itm_kl_loss = t.sum(F.log_softmax(all_int_embeds, dim=1) * F.softmax(int_embeds_m, dim=1), dim=1).mean()
        loss_itm = 0.4 * kd_int_2_loss - 0.6 * itm_kl_loss
        loss_itm *= self.kd_int_weight_3

        # ---  ITM Loss (Conformity) ---
        all_conf_embeds = t.cat([usrconf_embeds_proj, itmconf_embeds_proj], dim=0)
        usrconf_embeds_m = self.conf_mlp_m(self.usrconf_embeds)
        itmconf_embeds_m = self.conf_mlp_m(self.itmconf_embeds)
        conf_embeds_m = t.cat([usrconf_embeds_m, itmconf_embeds_m], dim=0)
        loss_itm_conf = t.sum(F.log_softmax(all_conf_embeds, dim=1) * F.softmax(conf_embeds_m, dim=1), dim=1).mean()
        loss_itm_conf *= -self.kd_conf_weight_3 # Maximize agreement -> Minimize negative loss

        # Final Loss
        loss = bpr_loss + reg_loss + kd_loss + kd_int_loss + kd_conf_loss + hcl_loss + loss_itm + loss_itm_conf
        losses = {
            'bpr_loss': bpr_loss, 
            'reg_loss': reg_loss, 
            'kd_loss': kd_loss, 
            'kd_int_loss': kd_int_loss, 
            'kd_conf_loss': kd_conf_loss,
            'hcl_loss': hcl_loss,
            'itm_loss': loss_itm,
            'itm_conf_loss': loss_itm_conf
        }
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _ = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds