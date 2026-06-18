import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.backends import cudnn
from Models import MainModel


class ProST:
    def __init__(self, adata, graph_dict, num_clusters, device, config, roundseed=0):
        seed = config['seed'] + roundseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.img = torch.as_tensor(adata.obsm['img_pca'], dtype=torch.float32, device=device)
        self.graph_dict = graph_dict
        self.mode = config['mode']
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters

        # Set default values if missing
        self.model_config.setdefault('use_img_adapter', True)
        self.model_config.setdefault('img_adapter_hidden', 256)
        self.model_config.setdefault('img_adapter_dropout', 0.1)
        self.model_config.setdefault('use_gate', True)
        self.model_config.setdefault('gate_hidden', 64)
        self.model_config.setdefault('gate_scale', 0.5)
        self.model_config.setdefault('use_positional_encoding', False)

        self.train_config['lr'] = float(self.train_config['lr'])
        self.train_config['decay'] = float(self.train_config['decay'])

    def _start_(self):
        if self.mode == 'clustering':
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        elif self.mode == 'imputation':
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Handle NaNs
        if torch.isnan(self.X).any():
            print("Warning: X contains NaN, replacing with 0")
            self.X = torch.nan_to_num(self.X, nan=0.0)
        if torch.isnan(self.img).any():
            print("Warning: img contains NaN, replacing with 0")
            self.img = torch.nan_to_num(self.img, nan=0.0)

        # Graph data
        if 'edge_index' in self.graph_dict and 'edge_type' in self.graph_dict:
            self.edge_index = self.graph_dict['edge_index'].long().to(self.device)
            self.edge_type = self.graph_dict['edge_type'].long().to(self.device)
            self.model_config.setdefault('graphtype', 'rgcn')
            if 'num_relations' in self.graph_dict:
                self.model_config.setdefault('num_relations', int(self.graph_dict['num_relations']))
            else:
                self.model_config.setdefault('num_relations', int(self.edge_type.max().item() + 1))
        else:
            self.edge_index = self.graph_dict['adj_label'].coalesce()._indices().to(self.device)
            self.edge_type = None
            self.model_config.setdefault('graphtype', 'gcn')

        self.input_dim = self.X.shape[-1]
        self.img_dim = self.img.shape[-1]
        self.num_nodes = self.X.shape[0]

        # Build model
        self.model = MainModel(
            self.num_clusters,
            self.input_dim,
            self.img_dim,
            self.model_config,
            self.device,
            num_nodes=self.num_nodes
        ).to(self.device)

        # Adjacency for contrastive learning (remove self-loops)
        mask = self.edge_index[0] != self.edge_index[1]
        edge_index_no_self = self.edge_index[:, mask]
        values = torch.ones(edge_index_no_self.size(1), device=self.device)
        adj_neigh = torch.sparse_coo_tensor(
            edge_index_no_self, values,
            (self.num_nodes, self.num_nodes)
        ).coalesce()
        self.model.adj_neigh = adj_neigh

        # Precompute attention mask
        self.model.cache_attention_mask(self.num_nodes, self.edge_index, self.device)

        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.train_config['lr'],
            weight_decay=self.train_config['decay'],
        )

    def _fit_(self):
        pbar = tqdm(range(self.train_config['epochs']))
        history = {
            'total': [], 'recon': [], 'contrast': [],
            'grad_norm': [],
            'enc_rep_std': [],
            'enc_rep_mean': [],
        }

        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad()

            contrast_loss, rec_loss = self.model(
                self.X, self.img, self.edge_index, edge_type=self.edge_type
            )
            loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_match'] * contrast_loss

            if torch.isnan(loss):
                print(f"NaN detected at epoch {epoch}, stopping training.")
                break

            loss.backward()

            total_grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            history['grad_norm'].append(total_grad_norm)

            if self.train_config['gradient_clipping'] > 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.train_config['gradient_clipping'])
            self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                rep, _ = self.model.evaluate(
                    self.X, self.img, self.edge_index, edge_type=self.edge_type
                )
                self.model.train()
                history['enc_rep_std'].append(rep.std().item())
                history['enc_rep_mean'].append(rep.mean().item())

            history['total'].append(loss.item())
            history['recon'].append(rec_loss.item())
            history['contrast'].append(contrast_loss.item())

            pbar.set_description(
                f"E{epoch} total={loss.item():.3f} rec={rec_loss.item():.3f} "
                f"cont={contrast_loss.item():.3f} grad={total_grad_norm:.2f} rep_std={rep.std().item():.3f}"
            )

        torch.cuda.empty_cache()
        return history

    def train(self):
        self._start_()
        self._fit_()

    def process(self):
        self.model.eval()
        enc_rep, recon = self.model.evaluate(
            self.X, self.img, self.edge_index, edge_type=self.edge_type
        )
        return enc_rep, recon