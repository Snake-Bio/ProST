import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch.backends import cudnn
from torch_geometric.nn import GCNConv


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True):
        super().__init__()
        self.conv = GCNConv(in_channels=in_features, out_channels=out_features)
        self.bn = nn.BatchNorm1d(out_features) if bn else nn.Identity()
        self.act = act
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        return x


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['latent_dim']
        self.gcn_hidden = config['project_dim']
        self.p_drop = config['p_drop']

        self.layer1 = GraphConv(self.input_dim, self.gcn_hidden,
                                dropout=self.p_drop, act=F.relu)
        self.layer2 = nn.Linear(self.gcn_hidden, self.input_dim, bias=False)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, config):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = config['latent_dim']
        self.p_drop = config['p_drop']

        self.layer1 = GraphConv(self.input_dim, self.output_dim,
                                dropout=self.p_drop, act=nn.Identity())

    def forward(self, x, edge_index):
        return self.layer1(x, edge_index)


class ProSTModel(nn.Module):
    def __init__(self, input_dim, img_dim, config, device):
        super().__init__()
        self.latent_dim = config['latent_dim']
        self.p_drop = config['p_drop']
        self.t = config['t']
        self.device = device

        self.gene_proj = nn.Linear(input_dim, self.latent_dim)
        self.img_proj = nn.Linear(img_dim, self.latent_dim)

        num_layers = config.get('transformer_layers', 4)
        num_heads = config.get('transformer_heads', 4)
        ff_dim = config.get('transformer_ff_dim', 512)
        dropout = config.get('transformer_dropout', 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.encoder_to_decoder = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.projector = Projector(config)
        self.decoder = Decoder(input_dim, config)

        self.sigmoid = nn.Sigmoid()
        self.adj_neigh = None

        self.mask_rate = config['mask_rate']
        self.replace_rate = config['replace_rate']
        self.mask_token_rate = 1 - self.replace_rate
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.rep_mask = nn.Parameter(torch.zeros(1, self.latent_dim))

        self.register_buffer('edge_index', None)

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[:int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        return out_x, mask_nodes

    def get_local_context(self, z, adj_neigh):
        neighbor_sum = torch.sparse.mm(adj_neigh, z)
        deg = torch.sparse.sum(adj_neigh, dim=1).to_dense().unsqueeze(-1) + 1e-8
        mean_neighbor = neighbor_sum / deg
        return self.sigmoid(mean_neighbor)

    def sce_loss(self, x, y, t=2, eps=1e-8):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
        cos_m = cos_m.clamp(min=eps, max=1-eps)
        loss = -torch.log(cos_m).pow(t)
        return loss.mean()

    def _build_attention_mask(self, N, edge_index, device):
        M = torch.ones(2*N, 2*N, dtype=torch.bool, device=device)
        M[range(2*N), range(2*N)] = False

        for i in range(N):
            gene_idx = N + i
            img_idx = i
            M[gene_idx, img_idx] = False

        adj = torch.zeros(N, N, dtype=torch.bool, device=device)
        adj[edge_index[0], edge_index[1]] = True
        adj = adj | adj.T
        for u in range(N):
            for v in torch.where(adj[u])[0]:
                M[N+u, N+v] = False

        return M

    def forward(self, x, img, edge_index, adj_coo, edge_type=None):
        N = x.shape[0]
        self.edge_index = edge_index

        use_x, mask_nodes = self.encoding_mask_noise(x, mask_rate=self.mask_rate)

        gene_tokens = self.gene_proj(use_x)
        img_tokens = self.img_proj(img)

        tokens = torch.cat([img_tokens, gene_tokens], dim=0).unsqueeze(0)

        attn_mask = self._build_attention_mask(N, edge_index, x.device)

        out = self.transformer(tokens, mask=attn_mask)
        out = out.squeeze(0)

        enc_rep = out[N:]

        if self.adj_neigh is not None and mask_nodes.numel() > 0:
            neighbor_avg_all = self.get_local_context(enc_rep, self.adj_neigh)
            anchor = enc_rep[mask_nodes]
            positive = neighbor_avg_all[mask_nodes]

            perm = torch.randperm(enc_rep.size(0), device=x.device)
            negative = enc_rep[perm[:mask_nodes.size(0)]]

            tau = 0.5
            pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / tau
            neg_sim = F.cosine_similarity(anchor, negative, dim=-1) / tau
            logits = torch.stack([pos_sim, neg_sim], dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=x.device)
            contrast_loss = F.cross_entropy(logits, labels)
        else:
            contrast_loss = torch.tensor(0.0, device=x.device)

        rep = self.encoder_to_decoder(enc_rep)
        rep[mask_nodes] = 0.0
        rep[mask_nodes] += self.rep_mask
        rep = self.projector(rep, edge_index)
        recon = self.decoder(rep, edge_index)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        rec_loss = self.sce_loss(x_rec, x_init, t=self.t)

        return contrast_loss, rec_loss

    @torch.no_grad()
    def evaluate(self, x, img, edge_index, adj_coo, edge_type=None):
        N = x.shape[0]

        gene_tokens = self.gene_proj(x)
        img_tokens = self.img_proj(img)
        tokens = torch.cat([img_tokens, gene_tokens], dim=0).unsqueeze(0)
        attn_mask = self._build_attention_mask(N, edge_index, x.device)

        out = self.transformer(tokens, mask=attn_mask).squeeze(0)
        enc_rep = out[N:]

        rep = self.encoder_to_decoder(enc_rep)
        rep = self.projector(rep, edge_index)
        recon = self.decoder(rep, edge_index)

        return rep, recon


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
        self.data_info = config['data']
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters

        self.model_config.setdefault('transformer_layers', 4)
        self.model_config.setdefault('transformer_heads', 4)
        self.model_config.setdefault('transformer_ff_dim', 512)
        self.model_config.setdefault('transformer_dropout', 0.1)

        self.train_config['lr'] = float(self.train_config['lr'])
        self.train_config['decay'] = float(self.train_config['decay'])

    def _start_(self):
        self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        print(f"X stats: mean={self.X.mean().item():.4f}, std={self.X.std().item():.4f}, "
              f"min={self.X.min().item():.4f}, max={self.X.max().item():.4f}")
        if torch.isnan(self.X).any():
            print("⚠️ X contains NaN, replacing with 0")
            self.X = torch.nan_to_num(self.X, nan=0.0)
        if torch.isnan(self.img).any():
            print("⚠️ img contains NaN, replacing with 0")
            self.img = torch.nan_to_num(self.img, nan=0.0)
        print(f"img stats: mean={self.img.mean().item():.4f}, std={self.img.std().item():.4f}, "
              f"min={self.img.min().item():.4f}, max={self.img.max().item():.4f}")

        if 'edge_index' in self.graph_dict and 'edge_type' in self.graph_dict:
            self.edge_index = self.graph_dict['edge_index'].long().to(self.device)
            self.edge_type = self.graph_dict['edge_type'].long().to(self.device)
            self.model_config['graphtype'] = 'gcn'
        else:
            self.edge_index = self.graph_dict['adj_label'].coalesce()._indices().to(self.device)
            self.edge_type = None

        readout_key = self.model_config.get('readout_key', 'adj_label')
        if readout_key in self.graph_dict:
            self.adj_coo = self.graph_dict[readout_key].to_sparse_coo().to(self.device)
        else:
            self.adj_coo = self.graph_dict['adj_label'].to_sparse_coo().to(self.device)

        self.norm_value = self.graph_dict.get('norm_value', 1.0)

        self.input_dim = self.X.shape[-1]
        self.img_dim = self.img.shape[-1]
        self.num_nodes = self.X.shape[0]

        self.model = ProSTModel(
            self.input_dim,
            self.img_dim,
            self.model_config,
            self.device
        ).to(self.device)

        mask = self.edge_index[0] != self.edge_index[1]
        edge_index_no_self = self.edge_index[:, mask]
        values = torch.ones(edge_index_no_self.size(1), device=self.device)
        adj_neigh = torch.sparse_coo_tensor(edge_index_no_self, values,
                                            (self.num_nodes, self.num_nodes)).coalesce()
        self.model.adj_neigh = adj_neigh

        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.train_config['lr'],
            weight_decay=self.train_config['decay'],
        )

    def _fit_(self):
        pbar = tqdm(range(self.train_config['epochs']))
        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad()

            contrast_loss, rec_loss = self.model(
                self.X, self.img, self.edge_index, self.adj_coo, edge_type=self.edge_type
            )
            loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_match'] * contrast_loss

            if torch.isnan(loss):
                print(f"NaN detected at epoch {epoch}, stopping training.")
                break

            loss.backward()
            if self.train_config.get('gradient_clipping', 0) > 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.train_config['gradient_clipping'])
            self.optimizer.step()

            pbar.set_description(
                f"Epoch {epoch} total loss={loss.item():.3f} "
                f"recon loss={rec_loss.item():.3f} contrast loss={contrast_loss.item():.3f}",
                refresh=True,
            )

        torch.cuda.empty_cache()

    def train(self):
        self._start_()
        self._fit_()

    def process(self):
        self.model.eval()
        enc_rep, recon = self.model.evaluate(
            self.X, self.img, self.edge_index, self.adj_coo, edge_type=self.edge_type
        )
        return enc_rep, recon