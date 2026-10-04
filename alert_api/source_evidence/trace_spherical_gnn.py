"""Tiny genuine icosahedral message-passing GNN for TRACE Stage 1.

Status: architecture + synthetic smoke test only. This does NOT count as
historical-weather validation. FANI must remain held out from training/tuning.

The module intentionally avoids torch_geometric so it can run with plain
PyTorch. It provides:
- true recursively subdivided icosphere mesh
- spherical grid<->mesh k-NN interpolation helpers
- edge-aware message passing
- lead-time-aware node features supplied by caller
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree


def _base_icosahedron():
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int64)
    return verts, faces


def build_icosphere(subdivisions=3):
    if subdivisions < 0 or subdivisions > 6:
        raise ValueError("subdivisions must be 0..6 for this prototype")
    verts, faces = _base_icosahedron()
    verts = [v.copy() for v in verts]
    faces = [tuple(map(int, f)) for f in faces]
    for _ in range(subdivisions):
        midpoint_cache = {}
        new_faces = []
        def midpoint(a, b):
            key = tuple(sorted((a,b)))
            if key in midpoint_cache:
                return midpoint_cache[key]
            v = (verts[a] + verts[b]) * 0.5
            v = v / np.linalg.norm(v)
            idx = len(verts); verts.append(v)
            midpoint_cache[key] = idx
            return idx
        for a,b,c in faces:
            ab, bc, ca = midpoint(a,b), midpoint(b,c), midpoint(c,a)
            new_faces += [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
        faces = new_faces
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    edges = set()
    for a,b,c in faces:
        for u,v in ((a,b),(b,c),(c,a)):
            edges.add((u,v)); edges.add((v,u))
    edge_index = np.asarray(sorted(edges), dtype=np.int64).T
    return verts, faces, edge_index


def xyz_to_latlon(xyz):
    xyz = np.asarray(xyz, dtype=float)
    lat = np.degrees(np.arcsin(np.clip(xyz[:,2], -1, 1)))
    lon = np.degrees(np.arctan2(xyz[:,1], xyz[:,0]))
    return lat, lon


def latlon_to_xyz(lat, lon):
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    return np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], axis=-1)


def knn_map(src_xyz, dst_xyz, k=3, power=2.0):
    """Return source indices + normalized inverse-distance weights for dst points."""
    src_xyz = np.asarray(src_xyz, dtype=float); dst_xyz = np.asarray(dst_xyz, dtype=float)
    tree = cKDTree(src_xyz)
    dist, idx = tree.query(dst_xyz, k=min(k, len(src_xyz)))
    if idx.ndim == 1:
        idx = idx[:,None]; dist = dist[:,None]
    w = 1.0 / np.maximum(dist, 1e-9)**power
    w /= w.sum(axis=1, keepdims=True)
    return idx.astype(np.int64), w.astype(np.float32)


def apply_knn(values, idx, weights):
    values = np.asarray(values)
    return np.sum(values[idx] * weights[...,None], axis=1) if values.ndim == 2 else np.sum(values[idx] * weights, axis=1)


class MessageBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden*2 + 4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, xyz, edge_index):
        src, dst = edge_index[0], edge_index[1]
        delta = xyz[src] - xyz[dst]
        chord = torch.linalg.norm(delta, dim=-1, keepdim=True)
        e = torch.cat([h[src], h[dst], delta, chord], dim=-1)
        msg = self.edge_mlp(e)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros((h.shape[0],1), device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones((dst.shape[0],1), device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        update = self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + update)


class TraceSphericalGNN(nn.Module):
    def __init__(self, in_features, hidden=64, blocks=3, out_features=1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([MessageBlock(hidden) for _ in range(blocks)])
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_features))

    def forward(self, node_features, mesh_xyz, edge_index):
        h = self.encoder(node_features)
        for block in self.blocks:
            h = block(h, mesh_xyz, edge_index)
        return self.decoder(h)


def synthetic_smoke_train(steps=25, subdivisions=2, seed=0):
    """Train on a synthetic spherical anomaly only to verify the full code path."""
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
    xyz_np, _, edges_np = build_icosphere(subdivisions)
    lat, lon = xyz_to_latlon(xyz_np)
    # Synthetic cyclone-shaped target near Bay of Bengal.
    center = latlon_to_xyz(np.array([16.0]), np.array([86.0]))[0]
    cosang = np.clip(xyz_np @ center, -1, 1)
    dist = np.arccos(cosang)
    signal = np.exp(-(dist/0.23)**2)
    lead = np.full_like(signal, 0.5)
    spread = 0.15 + 0.1*np.random.default_rng(seed).random(len(signal))
    features = np.stack([signal + 0.05*np.random.randn(len(signal)), spread, lead,
                         xyz_np[:,0], xyz_np[:,1], xyz_np[:,2]], axis=1).astype(np.float32)
    target = (signal > 0.35).astype(np.float32)[:,None]

    x = torch.from_numpy(features)
    y = torch.from_numpy(target)
    xyz = torch.from_numpy(xyz_np)
    edge_index = torch.from_numpy(edges_np)
    model = TraceSphericalGNN(in_features=x.shape[1], hidden=32, blocks=3)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    losses=[]
    for _ in range(steps):
        opt.zero_grad(); logits=model(x,xyz,edge_index); loss=loss_fn(logits,y); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    if not losses[-1] < losses[0]:
        raise AssertionError("synthetic GNN smoke training did not reduce loss")
    params=sum(p.numel() for p in model.parameters())
    return {"nodes":len(xyz_np),"directed_edges":edges_np.shape[1],"parameters":params,
            "loss_start":losses[0],"loss_end":losses[-1]}


if __name__ == "__main__":
    result = synthetic_smoke_train()
    print("SPHERICAL_GNN_SYNTHETIC_SMOKE_PASS", result)
