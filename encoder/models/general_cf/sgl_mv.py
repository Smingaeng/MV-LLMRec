import torch as t
from torch import nn
import numpy as np
import scipy.sparse as sp
import torch_sparse
import torch.nn.functional as F

from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.base_model import BaseModel
from models.model_utils import NodeDrop, SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class SGL_mv(BaseModel):
    def __init__(self, data_handler):
        super(SGL_mv, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.augmentation = configs['model']['augmentation']
        self.intent_num = configs['model']['intent_num']
        self.node_dropper = NodeDrop()
        self.edge_dropper = SpAdjEdgeDrop()
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.user_intent = t.nn.Parameter(init(t.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = t.nn.Parameter(init(t.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        
        # --- Conformity ---
        self.conformity_num = configs['model']['conformity_num'] 
        self.user_conformity = t.nn.Parameter(init(t.empty(self.embedding_size, self.conformity_num)), requires_grad=True)
        self.item_conformity = t.nn.Parameter(init(t.empty(self.embedding_size, self.conformity_num)), requires_grad=True)
        
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
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.kd_int_weight = self.hyper_config['kd_int_weight']
        self.kd_int_temperature = self.hyper_config['kd_int_temperature']
        
        # --- Conformity ---
        self.kd_conf_weight = self.hyper_config['kd_conf_weight']
        self.kd_conf_temperature = self.hyper_config['kd_conf_temperature']
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
        
        # --- Conformity Embeddings 및 MLP ---
        self.usrconf_embeds = t.tensor(configs['usrconf_embeds']).float().cuda()
        self.itmconf_embeds = t.tensor(configs['itmconf_embeds']).float().cuda()
        self.conf_mlp = nn.Sequential(
            nn.Linear(self.usrconf_embeds.shape[1], (self.usrconf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrconf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        # --- Gating Network ---
        self.gate_network = nn.Sequential(
            nn.Linear(self.embedding_size * 3, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, 3),
            nn.Softmax(dim=1)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear): init(m.weight)
        for m in self.int_mlp:
            if isinstance(m, nn.Linear): init(m.weight)
        for m in self.conf_mlp:
            if isinstance(m, nn.Linear): init(m.weight)
            
    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = t.nn.functional.normalize(head_embeddings)
        tail_embeddings = t.nn.functional.normalize(tail_embeddings)
        edge_alpha = (t.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(
            row=self.all_h_list, col=self.all_t_list,
            value=edge_alpha, sparse_sizes=self.A_in_shape
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

        # SGL Augmentation
        if self.is_training:
            if self.augmentation == 'node_drop':
                embeds = self.node_dropper(embeds, keep_rate)
            if self.augmentation == 'edge_drop':
                adj = self.edge_dropper(adj, keep_rate)
        
        embeds_list = [embeds]
        iaa_embeds_list = []
        caa_embeds_list = []

        for i in range(self.layer_num):
            # 1. Standard GCN propagation
            current_embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(current_embeds)

            # 2. Intent-aware Information Aggregation (IAA)
            u_embeds, i_embeds = t.split(embeds_list[i], [self.user_num, self.item_num], 0)
            u_int_embeds = t.softmax(u_embeds @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeds = t.softmax(i_embeds @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeds = t.concat([u_int_embeds, i_int_embeds], dim=0)
            head_embeds_int = t.index_select(int_layer_embeds, 0, self.all_h_list)
            tail_embeds_int = t.index_select(int_layer_embeds, 0, self.all_t_list)
            intent_indices, intent_values = self._adaptive_mask(head_embeds_int, tail_embeds_int)
            iaa_layer_embeds = torch_sparse.spmm(intent_indices, intent_values, self.A_in_shape[0], self.A_in_shape[1], embeds_list[i])
            iaa_embeds_list.append(iaa_layer_embeds)

            # 3. Conformity-aware Information Aggregation (CAA)
            u_conf_embeds = t.softmax(u_embeds @ self.user_conformity, dim=1) @ self.user_conformity.T
            i_conf_embeds = t.softmax(i_embeds @ self.item_conformity, dim=1) @ self.item_conformity.T
            conf_layer_embeds = t.concat([u_conf_embeds, i_conf_embeds], dim=0)
            head_embeds_conf = t.index_select(conf_layer_embeds, 0, self.all_h_list)
            tail_embeds_conf = t.index_select(conf_layer_embeds, 0, self.all_t_list)
            conformity_indices, conformity_values = self._adaptive_mask(head_embeds_conf, tail_embeds_conf)
            caa_layer_embeds = torch_sparse.spmm(conformity_indices, conformity_values, self.A_in_shape[0], self.A_in_shape[1], embeds_list[i])
            caa_embeds_list.append(caa_layer_embeds)

        gcn_embeds = sum(embeds_list)
        iaa_embeds = sum(iaa_embeds_list)
        caa_embeds = sum(caa_embeds_list)
        
        # --- Gating Adaptive Fusion---
        stacked_embeds = t.stack([gcn_embeds, iaa_embeds, caa_embeds], dim=1)
        flat_embeds = stacked_embeds.view(-1, self.embedding_size * 3)
        gate_weights = self.gate_network(flat_embeds).unsqueeze(-1)
        final_embeds = t.sum(stacked_embeds * gate_weights, dim=1)
        
        self.final_embeds = final_embeds
        disentangled_embeds = {'iaa': iaa_embeds, 'caa': caa_embeds, 'gcn': gcn_embeds}
        return final_embeds[:self.user_num], final_embeds[self.user_num:], disentangled_embeds

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        keep_rate = configs['model']['keep_rate']
        
        # Two augmented views for CL
        user_embeds1, item_embeds1, _ = self.forward(self.adj, keep_rate)
        user_embeds2, item_embeds2, _ = self.forward(self.adj, keep_rate)
        
        # One clean view for other losses
        user_embeds3, item_embeds3, disentangled_embeds = self.forward(self.adj, 1.0)
        
        ancs, poss, negs = batch_data
        anc_embeds1, pos_embeds1, _ = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, _ = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

        # BPR Loss & Regularization Loss (on clean view)
        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        
        # SGL Contrastive Loss (on augmented views)
        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_temperature)
        cl_loss = self.cl_weight * (cl_loss / anc_embeds1.shape[0])

        # KD Loss (Profile)
        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, _ = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)
        kd_loss = cal_infonce_loss(anc_embeds3, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds3, posprf_embeds, itmprf_embeds, self.kd_temperature)
        kd_loss = self.kd_weight * (kd_loss / anc_embeds3.shape[0])

        # KD Loss (Intent)
        iaa_embeds = disentangled_embeds['iaa']
        user_iaa_embeds, item_iaa_embeds = t.split(iaa_embeds, [self.user_num, self.item_num], 0)
        anc_iaa_embeds, pos_iaa_embeds, _ = self._pick_embeds(user_iaa_embeds, item_iaa_embeds, batch_data)
        usrint_embeds = self.int_mlp(self.usrint_embeds)
        itmint_embeds = self.int_mlp(self.itmint_embeds)
        ancint_embeds, posint_embeds, _ = self._pick_embeds(usrint_embeds, itmint_embeds, batch_data)
        kd_int_loss = cal_infonce_loss(anc_iaa_embeds, ancint_embeds, usrint_embeds, self.kd_int_temperature) + \
                      cal_infonce_loss(pos_iaa_embeds, posint_embeds, itmint_embeds, self.kd_int_temperature)
        kd_int_loss = self.kd_int_weight * (kd_int_loss / anc_embeds3.shape[0])

        # --- KD Loss (Conformity) ---
        caa_embeds = disentangled_embeds['caa']
        user_caa_embeds, item_caa_embeds = t.split(caa_embeds, [self.user_num, self.item_num], 0)
        anc_caa_embeds, pos_caa_embeds, _ = self._pick_embeds(user_caa_embeds, item_caa_embeds, batch_data)
        usrconf_embeds_proj = self.conf_mlp(self.usrconf_embeds)
        itmconf_embeds_proj = self.conf_mlp(self.itmconf_embeds)
        ancconf_embeds, posconf_embeds, _ = self._pick_embeds(usrconf_embeds_proj, itmconf_embeds_proj, batch_data)
        kd_conf_loss = cal_infonce_loss(anc_caa_embeds, ancconf_embeds, usrconf_embeds_proj, self.kd_conf_temperature) + \
                       cal_infonce_loss(pos_caa_embeds, posconf_embeds, itmconf_embeds_proj, self.kd_conf_temperature)
        kd_conf_loss = self.kd_conf_weight * (kd_conf_loss / anc_embeds3.shape[0])

        # --- Hierarchical Contrastive Learning ---
        gcn_embeds = disentangled_embeds['gcn']
        user_gcn_embeds, _ = t.split(gcn_embeds, [self.user_num, self.item_num], 0)
        anc_gcn_embeds = user_gcn_embeds[ancs]
        hcl_loss = (cal_infonce_loss(anc_gcn_embeds, anc_iaa_embeds, user_iaa_embeds, self.hcl_temperature) +
                    cal_infonce_loss(anc_gcn_embeds, anc_caa_embeds, user_caa_embeds, self.hcl_temperature) +
                    cal_infonce_loss(anc_iaa_embeds, anc_caa_embeds, user_caa_embeds, self.hcl_temperature))
        hcl_loss = self.hcl_weight * (hcl_loss / anc_embeds3.shape[0])
        
        # Final Loss
        loss = bpr_loss + reg_loss + cl_loss + kd_loss + kd_int_loss + kd_conf_loss + hcl_loss
        losses = {
            'bpr_loss': bpr_loss, 
            'reg_loss': reg_loss, 
            'cl_loss': cl_loss,
            'kd_loss': kd_loss, 
            'kd_int_loss': kd_int_loss, 
            'kd_conf_loss': kd_conf_loss,
            'hcl_loss': hcl_loss,
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