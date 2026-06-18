import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv, GATConv, RGCNConv


class GraphConv(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout=0.2,
        act=F.relu,
        bn=True,
        graphtype="gcn",
        num_relations=None,
        num_bases=None,
    ):
        super().__init__()
        bn_layer = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn_layer(out_features)
        self.act = act
        self.dropout = dropout
        self.graphtype = graphtype
        self.is_rgcn = False

        if graphtype == "gcn":
            self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gat":
            self.conv = GATConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gin":
            self.conv = TransformerConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "rgcn":
            if num_relations is None:
                raise ValueError("GraphConv(graphtype='rgcn') requires num_relations")
            self.conv = RGCNConv(
                in_channels=self.in_features,
                out_channels=self.out_features,
                num_relations=int(num_relations),
                num_bases=num_bases,
                root_weight=True,
                bias=True,
            )
            self.is_rgcn = True
        else:
            raise NotImplementedError(f"{graphtype} is not implemented.")

    def forward(self, x, edge_index, edge_type=None):
        if self.is_rgcn:
            if edge_type is None:
                edge_type = x.new_zeros(edge_index.size(1), dtype=torch.long)
            x = self.conv(x, edge_index, edge_type)
        else:
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
        graphtype = config.get('graphtype', 'gcn')
        num_relations = config.get('num_relations', None)
        num_bases = config.get('num_bases', None)

        self.layer1 = GraphConv(
            self.input_dim, self.gcn_hidden,
            dropout=self.p_drop, act=F.relu,
            graphtype=graphtype,
            num_relations=num_relations, num_bases=num_bases,
        )
        self.layer2 = nn.Linear(self.gcn_hidden, self.input_dim, bias=False)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x, edge_index, edge_type=None):
        x = self.layer1(x, edge_index, edge_type=edge_type)
        x = self.layer2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, config):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = config['latent_dim']
        self.p_drop = config['p_drop']
        graphtype = config.get('graphtype', 'gcn')
        num_relations = config.get('num_relations', None)
        num_bases = config.get('num_bases', None)

        self.layer1 = GraphConv(
            self.input_dim, self.output_dim,
            dropout=self.p_drop, act=nn.Identity(),
            graphtype=graphtype,
            num_relations=num_relations, num_bases=num_bases,
        )

    def forward(self, x, edge_index, edge_type=None):
        return self.layer1(x, edge_index, edge_type=edge_type)


class MainModel(nn.Module):
    def __init__(self, num_clusters, input_dim, img_dim, config, device, num_nodes=None):
        super().__init__()
        self.latent_dim = config['latent_dim']
        self.p_drop = config['p_drop']
        self.num_clusters = num_clusters
        self.device = device
        self.margin = config.get('margin', 0.0)
        self.power = config.get('power', 2)

        # Prompt pool
        self.use_prompt_pool = config.get('use_prompt_pool', True)
        self.prompt_inject_mode = config.get('prompt_inject_mode', 'add')
        if self.use_prompt_pool:
            self.pool_size = config.get('prompt_pool_size', 16)
            self.top_k = config.get('prompt_top_k', 3)
            self.prompt_temp = config.get('prompt_temperature', 2.0)
            self.prompt_keys = nn.Parameter(torch.empty(self.pool_size, self.latent_dim))
            self.prompt_values = nn.Parameter(torch.empty(self.pool_size, self.latent_dim))
            nn.init.xavier_uniform_(self.prompt_keys)
            nn.init.xavier_uniform_(self.prompt_values)
            if self.prompt_inject_mode == 'add':
                self.prompt_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.pool_size = 0
            self.top_k = 0
            self.prompt_temp = 1.0
            self.prompt_keys = None
            self.prompt_values = None

        # Projections
        self.gene_proj = nn.Linear(input_dim, self.latent_dim)
        self.img_proj = nn.Linear(img_dim, self.latent_dim)

        # Image adapter
        self.use_img_adapter = config.get('use_img_adapter', True)
        if self.use_img_adapter:
            adapter_hidden = config.get('img_adapter_hidden', 256)
            adapter_dropout = config.get('img_adapter_dropout', 0.1)
            self.img_adapter = nn.Sequential(
                nn.Linear(self.latent_dim, adapter_hidden),
                nn.GELU(),
                nn.Dropout(adapter_dropout),
                nn.Linear(adapter_hidden, self.latent_dim),
            )
            nn.init.constant_(self.img_adapter[-1].weight, 0)
            nn.init.constant_(self.img_adapter[-1].bias, 0)
        else:
            self.img_adapter = nn.Identity()

        # Transformer encoder
        num_layers = config.get('transformer_layers', 4)
        num_heads = config.get('transformer_heads', 4)
        ff_dim = config.get('transformer_ff_dim', 512)
        dropout = config.get('transformer_dropout', 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation='relu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Gated fusion
        self.use_gate = config.get('use_gate', True)
        if self.use_gate:
            gate_hidden = config.get('gate_hidden', 64)
            self.gate_scale = config.get('gate_scale', 0.5)
            self.gate_norm = nn.LayerNorm(self.latent_dim)
            self.gate_net = nn.Sequential(
                self.gate_norm, nn.Linear(self.latent_dim, gate_hidden),
                nn.ReLU(), nn.Linear(gate_hidden, 1), nn.Sigmoid()
            )
            nn.init.constant_(self.gate_net[-2].bias, -3.0)
        else:
            self.gate_scale = 1.0
            self.gate_net = None

        # Reconstruction path
        self.encoder_to_decoder = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.projector = Projector(config)
        self.decoder = Decoder(input_dim, config)

        self.sigmoid = nn.Sigmoid()
        self.register_buffer('adj_neigh', None)
        self.mask_rate = config['mask_rate']
        self.replace_rate = config['replace_rate']
        self.mask_token_rate = 1 - self.replace_rate
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.rep_mask = nn.Parameter(torch.zeros(1, self.latent_dim))

        # Positional encoding
        self.use_positional_encoding = config.get('use_positional_encoding', False)
        if self.use_positional_encoding:
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when use_positional_encoding=True")
            self.node_pos_embedding = nn.Parameter(torch.empty(num_nodes, self.latent_dim))
            nn.init.xavier_uniform_(self.node_pos_embedding)
        else:
            self.node_pos_embedding = None

        self.register_buffer('cached_attn_mask', None)

    def _build_attention_mask(self, N, edge_index, device):
        """Mask for 2N sequence (no pool or 'add' mode)"""
        M = torch.ones(2 * N, 2 * N, dtype=torch.bool, device=device)
        M[range(2 * N), range(2 * N)] = False
        idx_i = torch.arange(N, device=device)
        M[N + idx_i, idx_i] = False
        row, col = edge_index
        src = torch.cat([row, col])
        dst = torch.cat([col, row])
        M[N + src, N + dst] = False
        return M

    def _build_attention_mask_with_pool(self, N, edge_index, device):
        """Mask for 3N sequence ('concat' mode)"""
        total = 3 * N
        M = torch.ones(total, total, dtype=torch.bool, device=device)
        M[range(total), range(total)] = False
        idx_i = torch.arange(N, device=device)
        M[N + idx_i, idx_i] = False          # image -> gene
        M[2 * N + idx_i, idx_i] = False      # prompt -> gene
        M[2 * N + idx_i, N + idx_i] = False  # gene -> image
        row, col = edge_index
        src = torch.cat([row, col])
        dst = torch.cat([col, row])
        M[2 * N + src, 2 * N + dst] = False  # gene -> gene (neighbors)
        return M

    def cache_attention_mask(self, N, edge_index, device):
        if self.use_prompt_pool and self.prompt_inject_mode == 'concat':
            mask = self._build_attention_mask_with_pool(N, edge_index, device)
        else:
            mask = self._build_attention_mask(N, edge_index, device)
        self.register_buffer('cached_attn_mask', mask)

    def get_prompt_pool(self, img_tokens):
        N, D = img_tokens.shape
        q = F.normalize(img_tokens, p=2, dim=-1)
        k = F.normalize(self.prompt_keys, p=2, dim=-1)
        sim = torch.matmul(q, k.t()) / self.prompt_temp
        topk_sim, topk_idx = torch.topk(sim, self.top_k, dim=-1)
        topk_weight = F.softmax(topk_sim, dim=-1)
        topk_values = self.prompt_values[topk_idx]
        prompt_per_spot = torch.sum(topk_weight.unsqueeze(-1) * topk_values, dim=1)
        prompt_per_spot = F.layer_norm(prompt_per_spot, (D,))
        return prompt_per_spot

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

    def departure_loss(self, x, y, margin=0.0, power=2, eps=1e-8):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
        cos_m = cos_m.clamp(min=eps, max=1 - eps)
        loss = (-torch.log(cos_m)).pow(power).mean()
        return loss.mean()

    def forward(self, x, img, edge_index, edge_type=None):
        N = x.shape[0]
        use_x, mask_nodes = self.encoding_mask_noise(x, mask_rate=self.mask_rate)
        gene_tokens = self.gene_proj(use_x)
        img_features = self.img_proj(img)
        img_tokens = img_features + self.img_adapter(img_features)

        if self.use_positional_encoding:
            gene_tokens = gene_tokens + self.node_pos_embedding
            img_tokens = img_tokens + self.node_pos_embedding

        if self.use_prompt_pool:
            prompt = self.get_prompt_pool(img_tokens)
            if self.prompt_inject_mode == 'add':
                gene_tokens = gene_tokens + self.prompt_scale * prompt
                tokens = torch.cat([img_tokens, gene_tokens], dim=0).unsqueeze(0)
            else:  # concat
                tokens = torch.cat([prompt, img_tokens, gene_tokens], dim=0).unsqueeze(0)
        else:
            tokens = torch.cat([img_tokens, gene_tokens], dim=0).unsqueeze(0)

        attn_mask = self.cached_attn_mask if self.cached_attn_mask is not None else (
            self._build_attention_mask_with_pool(N, edge_index, x.device)
            if self.use_prompt_pool and self.prompt_inject_mode == 'concat'
            else self._build_attention_mask(N, edge_index, x.device)
        )
        out = self.transformer(tokens, mask=attn_mask).squeeze(0)

        if self.use_prompt_pool and self.prompt_inject_mode == 'concat':
            img_out = out[N:2 * N]
            gene_out = out[2 * N:]
        else:
            img_out = out[:N]
            gene_out = out[N:]

        if self.use_gate and self.gate_net is not None:
            gate = self.gate_net(img_out)
            enc_rep = gene_out + self.gate_scale * gate * img_out
        else:
            enc_rep = gene_out + self.gate_scale * img_out

        # Contrastive loss
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
        rep = self.projector(rep, edge_index, edge_type=edge_type)
        recon = self.decoder(rep, edge_index, edge_type=edge_type)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        rec_loss = self.departure_loss(x_rec, x_init, margin=self.margin, power=self.power)
        return contrast_loss, rec_loss

    @torch.no_grad()
    def evaluate(self, x, img, edge_index, edge_type=None):
        N = x.shape[0]
        gene_tokens = self.gene_proj(x)
        img_features = self.img_proj(img)
        img_tokens = img_features + self.img_adapter(img_features)

        if self.use_positional_encoding:
            gene_tokens = gene_tokens + self.node_pos_embedding
            img_tokens = img_tokens + self.node_pos_embedding

        if self.use_prompt_pool:
            prompt = self.get_prompt_pool(img_tokens)
            if self.prompt_inject_mode == 'add':
                gene_tokens = gene_tokens + self.prompt_scale * prompt
                tokens = torch.cat([img_tokens, gene_tokens], dim=0).unsqueeze(0)
            else:
                tokens = torch.cat([prompt, img_tokens, gene_tokens], dim=0).unsqueeze(0)
        else:
            tokens = torch.cat([img_tokens, gene_tokens], dim=0).unsqueeze(0)

        attn_mask = self.cached_attn_mask if self.cached_attn_mask is not None else (
            self._build_attention_mask_with_pool(N, edge_index, x.device)
            if self.use_prompt_pool and self.prompt_inject_mode == 'concat'
            else self._build_attention_mask(N, edge_index, x.device)
        )
        out = self.transformer(tokens, mask=attn_mask).squeeze(0)

        if self.use_prompt_pool and self.prompt_inject_mode == 'concat':
            img_out = out[N:2 * N]
            gene_out = out[2 * N:]
        else:
            img_out = out[:N]
            gene_out = out[N:]

        if self.use_gate and self.gate_net is not None:
            gate = self.gate_net(img_out)
            enc_rep = gene_out + self.gate_scale * gate * img_out
        else:
            enc_rep = gene_out + self.gate_scale * img_out

        rep = self.encoder_to_decoder(enc_rep)
        rep = self.projector(rep, edge_index, edge_type=edge_type)
        recon = self.decoder(rep, edge_index, edge_type=edge_type)
        return rep, recon