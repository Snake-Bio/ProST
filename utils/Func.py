import numpy as np
import torch
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import block_diag


def generate_adj_mat(adata, include_self: bool = False, n: int = 6):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'
    dist = metrics.pairwise_distances(adata.obsm['spatial'])
    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[: n + 1]
        adj_mat[i, n_neighbors] = 1
    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0
    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat


def generate_adj_mat_1(adata, max_dist):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'
    dist = metrics.pairwise_distances(adata.obsm['spatial'], metric='euclidean')
    adj_mat = dist < max_dist
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def mask_generator(adj_label, N: int = 1):
    idx = adj_label.indices()
    cell_num = adj_label.size()[0]
    list_non_neighbor = []
    for i in range(0, cell_num):
        neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
        n_selected = len(neighbor) * N
        total_idx = torch.arange(0, cell_num, dtype=torch.float32)
        non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
        indices = torch.randperm(len(non_neighbor), dtype=torch.float32)
        random_non_neighbor = indices[:n_selected]
        list_non_neighbor.append(random_non_neighbor)
    x = adj_label.indices()[0]
    y = torch.concat(list_non_neighbor)
    indices = torch.stack([x, y])
    indices = torch.concat([adj_label.indices(), indices], axis=1)
    value = torch.concat([adj_label.values(), torch.zeros(len(x), dtype=torch.float32)])
    adj_mask = torch.sparse_coo_tensor(indices, value)
    return adj_mask


def graph_computing(pos, n):
    from scipy.spatial import distance
    list_x = []
    list_y = []
    list_value = []
    for node_idx in range(len(pos)):
        tmp = pos[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, pos, 'euclidean')
        res = distMat.argsort()
        for j in np.arange(1, n + 1):
            list_x += [node_idx, res[0][j]]
            list_y += [res[0][j], node_idx]
            list_value += [1, 1]
    adj = sp.csr_matrix((list_value, (list_x, list_y)))
    adj = adj >= 1
    adj = adj.astype(np.float32)
    return adj


def graph_construction(adata, n: int = 6, dmax: float = 50, mode: str = 'KNN'):
    if mode == 'KNN':
        adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
    else:
        adj_m1 = generate_adj_mat_1(adata, dmax)
    adj_m1 = sp.coo_matrix(adj_m1)
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()
    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_m1 = adj_m1.tocoo()
    shape = adj_m1.shape
    values = adj_m1.data
    indices = np.stack([adj_m1.row, adj_m1.col])
    adj_label_m1 = torch.sparse_coo_tensor(indices, values, shape)
    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float(
        (adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2
    )
    graph_dict = {
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1.coalesce(),
        "norm_value": norm_m1,
    }
    return graph_dict


def _knn_coo(features: np.ndarray, k: int, metric: str = 'euclidean', include_self: bool = False, mode: str = 'connectivity') -> sp.coo_matrix:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    A = kneighbors_graph(features, n_neighbors=k, mode=mode, include_self=include_self, metric=metric)
    A = A.maximum(A.T)
    A = A.tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocoo()


def _filter_coo_by_spatial_radius(A: sp.coo_matrix, coords: np.ndarray, max_dist: float) -> sp.coo_matrix:
    if max_dist is None:
        return A
    if max_dist <= 0:
        return sp.coo_matrix(A.shape)
    row = A.row
    col = A.col
    diff = coords[row] - coords[col]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    mask = dist <= max_dist
    data = A.data[mask]
    row_f = row[mask]
    col_f = col[mask]
    return sp.coo_matrix((data, (row_f, col_f)), shape=A.shape)


def _coo_to_torch_sparse(A: sp.coo_matrix) -> torch.Tensor:
    A = A.tocoo()
    idx = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    val = torch.from_numpy(A.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, torch.Size(A.shape)).coalesce()


def _coo_to_edge_index(A: sp.coo_matrix) -> torch.Tensor:
    A = A.tocoo()
    return torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))


def graph_construction_multirel(
    adata,
    n_spatial: int = 6,
    n_gene: int = 6,
    n_img: int = 6,
    spatial_key: str = 'spatial',
    gene_key: str = 'X_pca',
    img_key: str = 'img_pca',
    constrain_gene_by_spatial: bool = True,
    constrain_img_by_spatial: bool = True,
    spatial_radius: float = 50,
    metric_gene: str = 'euclidean',
    metric_img: str = 'euclidean',
    include_self_in_readout: bool = True,
    rel_names=("spatial", "geneSim", "imgSim"),
):
    assert spatial_key in adata.obsm, f"AnnData.obsm must contain '{spatial_key}'"
    assert gene_key in adata.obsm, f"AnnData.obsm must contain '{gene_key}' (expression embedding, e.g. PCA)"
    assert img_key in adata.obsm, f"AnnData.obsm must contain '{img_key}' (image embedding, e.g. patch PCA)"
    coords = np.asarray(adata.obsm[spatial_key])
    X_gene = np.asarray(adata.obsm[gene_key])
    X_img = np.asarray(adata.obsm[img_key])
    N = coords.shape[0]
    A_spatial = _knn_coo(coords, k=n_spatial, metric='euclidean', include_self=False, mode='connectivity')
    A_gene = _knn_coo(X_gene, k=n_gene, metric=metric_gene, include_self=False, mode='connectivity')
    if constrain_gene_by_spatial:
        A_gene = _filter_coo_by_spatial_radius(A_gene, coords, spatial_radius)
    A_img = _knn_coo(X_img, k=n_img, metric=metric_img, include_self=False, mode='connectivity')
    if constrain_img_by_spatial:
        A_img = _filter_coo_by_spatial_radius(A_img, coords, spatial_radius)
    edge_index_list = []
    edge_type_list = []
    rel_mats = [A_spatial, A_gene, A_img]
    for rel_id, A in enumerate(rel_mats):
        A = A.tocoo()
        if A.nnz == 0:
            continue
        ei = _coo_to_edge_index(A)
        et = torch.full((ei.shape[1],), rel_id, dtype=torch.long)
        edge_index_list.append(ei)
        edge_type_list.append(et)
    if len(edge_index_list) == 0:
        raise RuntimeError("All relations are empty. Check your parameters (k/radius/keys).")
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_type = torch.cat(edge_type_list, dim=0)
    A_readout = A_spatial.tocsr().copy()
    if include_self_in_readout:
        A_readout = A_readout + sp.eye(N, format='csr')
    adj_label = _coo_to_torch_sparse(A_readout.tocoo())
    A_union = (A_spatial.tocsr() + A_gene.tocsr() + A_img.tocsr())
    A_union.data = np.ones_like(A_union.data)
    A_union[A_union > 0] = 1
    if include_self_in_readout:
        A_union = A_union + sp.eye(N, format='csr')
    adj_union = _coo_to_torch_sparse(A_union.tocoo())
    adj_norm = preprocess_graph(A_spatial)
    norm_value = N * N / float((N * N - A_readout.sum()) * 2)
    graph_dict = {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_relations": int(len(rel_names)),
        "rel_names": list(rel_names),
        "adj_norm": adj_norm,
        "adj_label": adj_label,
        "adj_union": adj_union,
        "norm_value": norm_value,
        "edge_counts": {
            "spatial": int(A_spatial.nnz),
            "geneSim": int(A_gene.nnz),
            "imgSim": int(A_img.nnz),
            "total": int(edge_index.size(1)),
        },
    }
    return graph_dict


def coo2csr(coo_matrix_t: torch.Tensor):
    coo_matrix_t = coo_matrix_t.coalesce()
    indices = coo_matrix_t.indices()
    values = coo_matrix_t.values()
    sparse_matrix = sp.coo_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=coo_matrix_t.size())
    csr_matrix = sparse_matrix.tocsr()
    return csr_matrix


def csr2coo(csr_matrix: sp.csr_matrix):
    coo = csr_matrix.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    size = torch.Size(coo.shape)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size)
    return sparse_tensor.coalesce()


def combine_graph_dict_1(dict_1, dict_2):
    tmp_adj_norm = csr2coo(block_diag([coo2csr(dict_1['adj_norm']), coo2csr(dict_2['adj_norm'])]))
    tmp_adj_label = csr2coo(block_diag([coo2csr(dict_1['adj_label']), coo2csr(dict_2['adj_label'])]))
    graph_dict = {
        "adj_norm": tmp_adj_norm,
        "adj_label": tmp_adj_label,
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']]),
    }
    return graph_dict


def combine_graph_dict(dict_1, dict_2):
    if ('edge_index' in dict_1) and ('edge_index' in dict_2):
        n1 = dict_1['adj_label'].size(0)
        n2 = dict_2['adj_label'].size(0)
        edge_index = torch.cat(
            [
                dict_1['edge_index'],
                dict_2['edge_index'] + n1,
            ],
            dim=1,
        )
        edge_type = torch.cat([dict_1['edge_type'], dict_2['edge_type']], dim=0)
        adj_label = csr2coo(block_diag([coo2csr(dict_1['adj_label']), coo2csr(dict_2['adj_label'])]))
        adj_union = None
        if ('adj_union' in dict_1) and ('adj_union' in dict_2):
            adj_union = csr2coo(block_diag([coo2csr(dict_1['adj_union']), coo2csr(dict_2['adj_union'])]))
        graph_dict = {
            "edge_index": edge_index,
            "edge_type": edge_type,
            "num_relations": dict_1.get('num_relations', int(edge_type.max().item() + 1)),
            "rel_names": dict_1.get('rel_names', None),
            "adj_label": adj_label,
            "norm_value": np.mean([dict_1.get('norm_value', 1.0), dict_2.get('norm_value', 1.0)]),
        }
        if adj_union is not None:
            graph_dict['adj_union'] = adj_union
        return graph_dict
    tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
    tmp_adj_norm = tmp_adj_norm.to_sparse()
    tmp_adj_label = torch.block_diag(dict_1['adj_label'].to_dense(), dict_2['adj_label'].to_dense())
    tmp_adj_label = tmp_adj_label.to_sparse()
    graph_dict = {
        "adj_norm": tmp_adj_norm,
        "adj_label": tmp_adj_label,
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']]),
    }
    return graph_dict