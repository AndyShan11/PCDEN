"""
消融实验: 诊断 NM-NBFNet 性能低下的根因
============================================
目的: 区分两种假设:
  假设A: "非马尔可夫修改"本身有害 → 修复方向: 重新设计非马尔可夫机制
  假设B: "模型太小+训练不稳定"   → 修复方向: 增大dim, 调优超参

实验设计 (2x2 消融):
  ┌──────────────┬──────────┬──────────┐
  │              │ dim=32   │ dim=64   │
  ├──────────────┼──────────┼──────────┤
  │ Vanilla NBF  │ 对照基线 │ 容量测试 │
  │ NM-NBFNet    │ 原始结果 │ 容量测试 │
  └──────────────┴──────────┴──────────┘

控制变量: 相同的数据、相同的训练流程、相同的评估方式

用法:
  python ablation_experiment.py                    # 完整 2x2 消融
  python ablation_experiment.py --configs vanilla_32 nm_64  # 只跑指定配置
  python ablation_experiment.py --epochs 10        # 指定epoch数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, time, random, math, argparse, json
import urllib.request
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "ablation_log.txt")
_log_f = open(LOG_FILE, "w", encoding="utf-8")

def log(msg=""):
    print(msg, flush=True)
    _log_f.write(msg + "\n")
    _log_f.flush()

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"[Device] {device}")
if device.type == "cuda":
    log(f"[GPU] {torch.cuda.get_device_name(0)}")
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log(f"[GPU Memory] {gpu_mem_gb:.1f} GB")
else:
    gpu_mem_gb = 0

DATA_DIR = os.path.join(BASE_DIR, "data", "FB15k-237")


# =================================================================
# 1. 数据集 (复用)
# =================================================================

def ensure_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(os.path.join(DATA_DIR, "train.txt")):
        return
    log("[Data] 下载 FB15k-237...")
    base = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237"
    for s in ["train", "valid", "test"]:
        urllib.request.urlretrieve(f"{base}/{s}.txt", os.path.join(DATA_DIR, f"{s}.txt"))
    log("[Data] 下载完成")


def load_data():
    ensure_dataset()
    ent2id, rel2id = {}, {}
    def _read(path):
        triples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) != 3: continue
                h, r, t = p
                if h not in ent2id: ent2id[h] = len(ent2id)
                if r not in rel2id: rel2id[r] = len(rel2id)
                if t not in ent2id: ent2id[t] = len(ent2id)
                triples.append((ent2id[h], rel2id[r], ent2id[t]))
        return triples
    train = _read(os.path.join(DATA_DIR, "train.txt"))
    valid = _read(os.path.join(DATA_DIR, "valid.txt"))
    test  = _read(os.path.join(DATA_DIR, "test.txt"))
    nE, nR = len(ent2id), len(rel2id)
    log(f"[Data] 实体={nE}, 关系={nR}, train={len(train)}, valid={len(valid)}, test={len(test)}")
    return train, valid, test, nE, nR


class KGGraph:
    def __init__(self, triples, nE, nR):
        self.nE, self.nR = nE, nR
        self.total_rel = nR * 2
        self.adj = defaultdict(list)
        for h, r, t in triples:
            self.adj[h].append((t, r))
            self.adj[t].append((h, r + nR))
        src, tgt, rel = [], [], []
        for h, r, t in triples:
            src.append(h); tgt.append(t); rel.append(r)
            src.append(t); tgt.append(h); rel.append(r + nR)
        self.full_src = torch.tensor(src, dtype=torch.long)
        self.full_tgt = torch.tensor(tgt, dtype=torch.long)
        self.full_rel = torch.tensor(rel, dtype=torch.long)
        self.num_edges = len(src)
        self.true_tails = defaultdict(lambda: defaultdict(set))
        for h, r, t in triples:
            self.true_tails[h][r].add(t)
            self.true_tails[t][r + nR].add(h)

    def add_eval_triples(self, valid_triples, test_triples):
        for triples in [valid_triples, test_triples]:
            for h, r, t in triples:
                self.true_tails[h][r].add(t)
                self.true_tails[t][r + self.nR].add(h)

    def sample_subgraph(self, max_edges_per_node=30):
        src, tgt, rel = [], [], []
        for node in range(self.nE):
            neighbors = self.adj[node]
            if len(neighbors) <= max_edges_per_node:
                sample = neighbors
            else:
                sample = random.sample(neighbors, max_edges_per_node)
            for nb, r in sample:
                src.append(node); tgt.append(nb); rel.append(r)
        return (torch.tensor(src, dtype=torch.long, device=device),
                torch.tensor(tgt, dtype=torch.long, device=device),
                torch.tensor(rel, dtype=torch.long, device=device))


# =================================================================
# 2. 两种模型: Vanilla NBFNet vs NM-NBFNet
# =================================================================

class VanillaBellmanFordLayer(nn.Module):
    """标准 Bellman-Ford 层 — 无门控, 简单消息传递 (对照组)"""
    def __init__(self, hidden_dim, num_relations):
        super().__init__()
        self.msg_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_src, edge_tgt, edge_rel, nE):
        batch_size, _, dim = h.shape
        r = self.rel_emb(edge_rel)                   # [E, dim]
        h_src = h[:, edge_src]                        # [B, E, dim]
        msg = self.msg_linear(h_src) * r.unsqueeze(0) # [B, E, dim]
        idx = edge_tgt.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, dim)
        h_agg = torch.zeros(batch_size, nE, dim, device=h.device)
        h_agg.scatter_add_(1, idx, msg)
        h_new = self.norm(h + F.relu(self.update(torch.cat([h, h_agg], dim=-1))))
        return h_new


class NMBellmanFordLayer(nn.Module):
    """非马尔可夫 Bellman-Ford 层 — 带门控机制 (实验组)"""
    def __init__(self, hidden_dim, num_relations):
        super().__init__()
        self.msg_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.gate_w = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_src, edge_tgt, edge_rel, nE):
        batch_size, _, dim = h.shape
        r = self.rel_emb(edge_rel)
        h_src = h[:, edge_src]
        gate = torch.sigmoid(self.gate_w(h_src))      # 非马尔可夫门控
        msg = self.msg_linear(h_src) * r.unsqueeze(0) * gate
        idx = edge_tgt.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, dim)
        h_agg = torch.zeros(batch_size, nE, dim, device=h.device)
        h_agg.scatter_add_(1, idx, msg)
        h_new = self.norm(h + F.relu(self.update(torch.cat([h, h_agg], dim=-1))))
        return h_new


class NBFNet(nn.Module):
    """通用 NBFNet — 通过 variant 参数选择 vanilla 或 non-markov"""

    def __init__(self, nE, nR, dim=32, num_layers=3, dropout=0.1, variant="vanilla"):
        super().__init__()
        self.nE = nE
        self.dim = dim
        self.variant = variant
        total_rel = nR * 2
        self.ent_emb = nn.Embedding(nE, dim)
        self.query_emb = nn.Embedding(total_rel, dim)

        LayerClass = NMBellmanFordLayer if variant == "nm" else VanillaBellmanFordLayer
        self.layers = nn.ModuleList([LayerClass(dim, total_rel) for _ in range(num_layers)])

        self.score_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 1)
        )
        self._init()

    def _init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source, query_rel, edge_src, edge_tgt, edge_rel, targets=None):
        bs = source.shape[0]
        h = torch.zeros(bs, self.nE, self.dim, device=source.device)
        init = self.ent_emb(source) + self.query_emb(query_rel)
        h.scatter_(1, source.view(bs, 1, 1).expand(-1, 1, self.dim), init.unsqueeze(1))

        for layer in self.layers:
            h = layer(h, edge_src, edge_tgt, edge_rel, self.nE)

        if targets is not None:
            K = targets.shape[1]
            bi = torch.arange(bs, device=h.device).unsqueeze(1).expand(-1, K)
            ht = h[bi, targets]
            te = self.ent_emb(targets)
            return self.score_mlp(torch.cat([ht, te], -1)).squeeze(-1)
        else:
            te = self.ent_emb.weight.unsqueeze(0).expand(bs, -1, -1)
            return self.score_mlp(torch.cat([h, te], -1)).squeeze(-1)


# =================================================================
# 3. 训练 & 评估
# =================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            ratio = self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            ratio = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * ratio


def train_epoch(model, train_data, graph, optimizer, nE, batch_size=4, neg_size=128,
                max_steps=5000, edges_per_node=30, scaler=None, accum_steps=4,
                adv_temp=5.0, resample_interval=2000, scheduler=None):
    model.train()
    random.shuffle(train_data)
    total_loss, n = 0, 0
    use_amp = (device.type == "cuda" and scaler is not None)

    e_src, e_tgt, e_rel = graph.sample_subgraph(max_edges_per_node=edges_per_node)
    optimizer.zero_grad()
    step_in_accum = 0
    global_step = 0

    total_steps = min(len(train_data), max_steps * batch_size)
    for i in range(0, total_steps, batch_size):
        batch = train_data[i:i + batch_size]
        if not batch:
            continue

        global_step += 1
        if resample_interval > 0 and global_step % resample_interval == 0:
            e_src, e_tgt, e_rel = graph.sample_subgraph(max_edges_per_node=edges_per_node)

        heads = torch.tensor([x[0] for x in batch], device=device)
        rels  = torch.tensor([x[1] for x in batch], device=device)
        tails = torch.tensor([x[2] for x in batch], device=device)
        bs = len(batch)

        neg = torch.randint(0, nE, (bs, neg_size), device=device)
        targets = torch.cat([tails.unsqueeze(1), neg], dim=1)

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                scores = model(heads, rels, e_src, e_tgt, e_rel, targets=targets)
                pos_s, neg_s = scores[:, 0], scores[:, 1:]
                pos_loss = -F.logsigmoid(pos_s).mean()
                with torch.no_grad():
                    w = F.softmax(neg_s * adv_temp, dim=1)
                neg_loss = -(w * F.logsigmoid(-neg_s)).sum(1).mean()
                loss = (pos_loss + neg_loss) / (2 * accum_steps)
            scaler.scale(loss).backward()
        else:
            scores = model(heads, rels, e_src, e_tgt, e_rel, targets=targets)
            pos_s, neg_s = scores[:, 0], scores[:, 1:]
            pos_loss = -F.logsigmoid(pos_s).mean()
            with torch.no_grad():
                w = F.softmax(neg_s * adv_temp, dim=1)
            neg_loss = -(w * F.logsigmoid(-neg_s)).sum(1).mean()
            loss = (pos_loss + neg_loss) / (2 * accum_steps)
            loss.backward()

        step_in_accum += 1
        total_loss += loss.item() * accum_steps
        n += 1

        if step_in_accum >= accum_steps:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            step_in_accum = 0
            if scheduler is not None:
                scheduler.step()

    # 处理剩余的梯度
    if step_in_accum > 0:
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, triples, graph, nE, max_test=1000, batch_eval=4):
    model.eval()
    subset = triples[:max_test]
    ranks = []

    e_src = graph.full_src.to(device)
    e_tgt = graph.full_tgt.to(device)
    e_rel = graph.full_rel.to(device)

    for start in range(0, len(subset), batch_eval):
        batch = subset[start:start + batch_eval]
        bs = len(batch)
        heads = torch.tensor([x[0] for x in batch], device=device)
        rels  = torch.tensor([x[1] for x in batch], device=device)
        scores = model(heads, rels, e_src, e_tgt, e_rel)

        for j in range(bs):
            hh, rr, tt = batch[j]
            s = scores[j].clone()
            ts = s[tt].item()
            for e in graph.true_tails[hh][rr]:
                if e != tt:
                    s[e] = float('-inf')
            rank = (s > ts).sum().item() + 1
            ranks.append(rank)

    ranks = np.array(ranks, dtype=float)
    mrr  = np.mean(1.0 / ranks)
    h1   = np.mean(ranks <= 1) * 100
    h3   = np.mean(ranks <= 3) * 100
    h10  = np.mean(ranks <= 10) * 100
    return mrr, h1, h3, h10


# =================================================================
# 4. 消融实验配置
# =================================================================

def get_ablation_configs(gpu_mem_gb, args):
    """
    根据GPU显存自动调整配置.
    返回 {config_name: {params...}} 字典.
    """
    configs = {}

    # ---- dim=32 配置 (任何GPU都能跑) ----
    base_32 = dict(
        dim=32, num_layers=3, lr=5e-4, batch_size=4, accum_steps=4,
        neg_size=128, edges_per_node=30, epochs=args.epochs,
        steps_per_epoch=args.steps, dropout=0.1, adv_temp=5.0,
    )
    configs["vanilla_32"] = {**base_32, "variant": "vanilla", "label": "Vanilla-NBFNet dim=32"}
    configs["nm_32"]      = {**base_32, "variant": "nm",      "label": "NM-NBFNet dim=32"}

    # ---- dim=64 配置 (根据显存调整batch) ----
    if gpu_mem_gb >= 10:
        bs64, acc64, neg64, epn64 = 4, 4, 128, 30
    elif gpu_mem_gb >= 6:
        bs64, acc64, neg64, epn64 = 2, 8, 64, 20
    else:
        # CPU 或极小显存
        bs64, acc64, neg64, epn64 = 1, 16, 32, 15

    base_64 = dict(
        dim=64, num_layers=3, lr=3e-4, batch_size=bs64, accum_steps=acc64,
        neg_size=neg64, edges_per_node=epn64, epochs=args.epochs,
        steps_per_epoch=args.steps, dropout=0.15, adv_temp=5.0,
    )
    configs["vanilla_64"] = {**base_64, "variant": "vanilla", "label": "Vanilla-NBFNet dim=64"}
    configs["nm_64"]      = {**base_64, "variant": "nm",      "label": "NM-NBFNet dim=64"}

    return configs


def run_single_config(config_name, cfg, train_data, valid, test, graph, nE, nR):
    """运行单个配置, 返回结果字典"""
    log(f"\n{'='*60}")
    log(f" 消融配置: {cfg['label']}")
    log(f"{'='*60}")

    # 每次重新设随机种子, 确保公平比较
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = NBFNet(nE, nR, dim=cfg['dim'], num_layers=cfg['num_layers'],
                   dropout=cfg['dropout'], variant=cfg['variant']).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    log(f"  参数量: {nparams:,}")
    log(f"  batch={cfg['batch_size']}x{cfg['accum_steps']}(accum)={cfg['batch_size']*cfg['accum_steps']}, "
        f"neg={cfg['neg_size']}, edges/node={cfg['edges_per_node']}, lr={cfg['lr']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    steps_per_epoch_optim = cfg['steps_per_epoch'] // cfg['accum_steps']
    total_training_steps = steps_per_epoch_optim * cfg['epochs']
    warmup_steps = steps_per_epoch_optim  # 1 epoch warmup
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_training_steps)

    best_mrr = 0
    patience_counter = 0
    patience = 5
    save_path = os.path.join(DATA_DIR, f"ablation_{config_name}.pt")

    history = {'epochs': [], 'losses': [], 'val_mrrs': [], 'val_h1s': [], 'val_h3s': [], 'val_h10s': []}
    t0 = time.time()

    for epoch in range(1, cfg['epochs'] + 1):
        te = time.time()

        loss = train_epoch(
            model, train_data, graph, optimizer, nE,
            batch_size=cfg['batch_size'], neg_size=cfg['neg_size'],
            max_steps=cfg['steps_per_epoch'],
            edges_per_node=cfg['edges_per_node'],
            scaler=scaler, accum_steps=cfg['accum_steps'],
            adv_temp=cfg['adv_temp'], resample_interval=2000,
            scheduler=scheduler,
        )

        ep_time = time.time() - te
        mrr, h1, h3, h10 = evaluate(model, valid, graph, nE, max_test=500, batch_eval=4)

        history['epochs'].append(epoch)
        history['losses'].append(loss)
        history['val_mrrs'].append(mrr)
        history['val_h1s'].append(h1)
        history['val_h3s'].append(h3)
        history['val_h10s'].append(h10)

        marker = ""
        if mrr > best_mrr:
            best_mrr = mrr
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            marker = " ★"
        else:
            patience_counter += 1

        log(f"  Epoch {epoch:2d} | loss={loss:.4f} | MRR={mrr:.4f} H@1={h1:.1f} H@10={h10:.1f} | {ep_time:.0f}s{marker}")

        if patience_counter >= patience:
            log(f"  Early stopping at epoch {epoch}")
            break

    # 测试
    train_time = (time.time() - t0) / 60
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    test_mrr, test_h1, test_h3, test_h10 = evaluate(model, test, graph, nE, max_test=2000, batch_eval=4)

    log(f"  Test: MRR={test_mrr:.4f} H@1={test_h1:.1f}% H@3={test_h3:.1f}% H@10={test_h10:.1f}%")
    log(f"  训练时间: {train_time:.1f}分钟, 参数量: {nparams:,}")

    # 清理GPU
    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        'config_name': config_name,
        'label': cfg['label'],
        'variant': cfg['variant'],
        'dim': cfg['dim'],
        'nparams': nparams,
        'train_time_min': train_time,
        'best_val_mrr': best_mrr,
        'test_mrr': test_mrr,
        'test_h1': test_h1,
        'test_h3': test_h3,
        'test_h10': test_h10,
        'history': history,
    }


# =================================================================
# 5. 结果分析与可视化
# =================================================================

def analyze_and_plot(all_results):
    """生成消融实验分析图和结论"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matplotlib.rcParams['font.size'] = 11
    except ImportError:
        log("[Plot] matplotlib 未安装，跳过绘图")
        return

    fig_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 按 dim 和 variant 组织结果
    results_map = {r['config_name']: r for r in all_results}

    # ================================================================
    # 图1: 训练曲线对比 (Loss)
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'vanilla_32': '#1f77b4', 'nm_32': '#d62728',
        'vanilla_64': '#2ca02c', 'nm_64': '#ff7f0e',
    }
    markers = {
        'vanilla_32': 'o', 'nm_32': 's',
        'vanilla_64': '^', 'nm_64': 'D',
    }

    for name, r in results_map.items():
        c, m = colors.get(name, 'gray'), markers.get(name, 'o')
        ax1.plot(r['history']['epochs'], r['history']['losses'],
                 f'-{m}', color=c, lw=2, ms=5, label=r['label'])
        ax2.plot(r['history']['epochs'], r['history']['val_mrrs'],
                 f'-{m}', color=c, lw=2, ms=5, label=r['label'])

    # SOTA 参考线
    for name, val, ls in [("NBFNet", 0.415, '--'), ("TransE", 0.294, ':')]:
        ax2.axhline(y=val, color='gray', ls=ls, alpha=0.5, label=f"{name} ({val})")

    ax1.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax2.set(xlabel="Epoch", ylabel="MRR", title="Validation MRR")
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=9, loc='lower right'); ax2.grid(True, alpha=0.3)

    fig.suptitle("Ablation: Training Curves Comparison", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "ablation_training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  [Plot] -> figures/ablation_training_curves.png")

    # ================================================================
    # 图2: 2x2 消融结果柱状图
    # ================================================================
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    metrics = [('test_mrr', 'MRR'), ('test_h1', 'Hits@1 (%)'), ('test_h3', 'Hits@3 (%)'), ('test_h10', 'Hits@10 (%)')]

    for ax, (key, title) in zip(axes, metrics):
        labels, vals, bar_colors = [], [], []
        for name in ['vanilla_32', 'nm_32', 'vanilla_64', 'nm_64']:
            if name in results_map:
                r = results_map[name]
                labels.append(r['label'].replace(' dim=', '\ndim='))
                vals.append(r[key])
                bar_colors.append(colors[name])

        bars = ax.bar(range(len(labels)), vals, color=bar_colors, alpha=0.85, edgecolor='white')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        for idx, v in enumerate(vals):
            fmt = f'{v:.4f}' if key == 'test_mrr' else f'{v:.1f}'
            ax.text(idx, v + max(vals)*0.02, fmt, ha='center', fontsize=9, fontweight='bold')

    fig.suptitle("Ablation: Test Performance (2x2 Design)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "ablation_bar_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  [Plot] -> figures/ablation_bar_comparison.png")

    # ================================================================
    # 图3: 交互效应分析 (Dim x Variant)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    dims = [32, 64]

    vanilla_mrrs = [results_map.get(f'vanilla_{d}', {}).get('test_mrr', 0) for d in dims]
    nm_mrrs      = [results_map.get(f'nm_{d}', {}).get('test_mrr', 0) for d in dims]

    ax.plot(dims, vanilla_mrrs, 'o-', color='#1f77b4', lw=2.5, ms=10, label='Vanilla NBFNet')
    ax.plot(dims, nm_mrrs,      's-', color='#d62728', lw=2.5, ms=10, label='NM-NBFNet')

    # 标注差值
    for d_idx, d in enumerate(dims):
        v_mrr, nm_mrr = vanilla_mrrs[d_idx], nm_mrrs[d_idx]
        if v_mrr > 0 and nm_mrr > 0:
            diff = nm_mrr - v_mrr
            mid = (v_mrr + nm_mrr) / 2
            sign = '+' if diff >= 0 else ''
            ax.annotate(f'Δ={sign}{diff:.4f}', xy=(d, mid), fontsize=10,
                        fontweight='bold', color='purple', ha='left',
                        xytext=(d+2, mid))

    ax.axhline(y=0.415, color='green', ls='--', alpha=0.5, label='NBFNet target (0.415)')
    ax.set_xticks(dims)
    ax.set(xlabel="Embedding Dimension", ylabel="Test MRR",
           title="Interaction Effect: Dimension × Variant")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "ablation_interaction.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  [Plot] -> figures/ablation_interaction.png")


def print_diagnosis(all_results):
    """基于消融结果输出诊断结论"""
    r = {res['config_name']: res for res in all_results}

    log("\n" + "=" * 70)
    log(" 消融实验诊断结论")
    log("=" * 70)

    # 汇总表
    log(f"\n{'Config':<25} {'Params':>10} {'Val MRR':>10} {'Test MRR':>10} {'H@1':>7} {'H@10':>7} {'Time':>8}")
    log("-" * 80)
    for name in ['vanilla_32', 'nm_32', 'vanilla_64', 'nm_64']:
        if name in r:
            res = r[name]
            log(f"  {res['label']:<23} {res['nparams']:>10,} {res['best_val_mrr']:>10.4f} "
                f"{res['test_mrr']:>10.4f} {res['test_h1']:>6.1f}% {res['test_h10']:>6.1f}% "
                f"{res['train_time_min']:>6.1f}m")

    # 参考
    log(f"\n  参考: NBFNet (论文) = MRR 0.415, TransE = 0.294")

    # 诊断逻辑
    v32 = r.get('vanilla_32', {}).get('test_mrr', 0)
    nm32 = r.get('nm_32', {}).get('test_mrr', 0)
    v64 = r.get('vanilla_64', {}).get('test_mrr', 0)
    nm64 = r.get('nm_64', {}).get('test_mrr', 0)

    log(f"\n  [分析]")

    if v32 > 0 and nm32 > 0:
        gap32 = v32 - nm32
        log(f"  dim=32 差距 (Vanilla - NM): {gap32:+.4f}")

    if v64 > 0 and nm64 > 0:
        gap64 = v64 - nm64
        log(f"  dim=64 差距 (Vanilla - NM): {gap64:+.4f}")

    if v32 > 0 and v64 > 0:
        v_gain = v64 - v32
        log(f"  Vanilla提升 (64 vs 32):     {v_gain:+.4f}")

    if nm32 > 0 and nm64 > 0:
        nm_gain = nm64 - nm32
        log(f"  NM提升 (64 vs 32):          {nm_gain:+.4f}")

    log(f"\n  [结论]")

    if v32 > 0 and nm32 > 0 and v64 > 0 and nm64 > 0:
        gap32 = v32 - nm32
        gap64 = v64 - nm64

        if gap32 > 0.05 and gap64 > 0.05:
            log("  → 假设A成立: 非马尔可夫门控机制本身有害")
            log("    无论 dim=32 还是 dim=64, Vanilla 均显著优于 NM")
            log("    建议: 重新设计非马尔可夫机制 (如路径记忆、注意力历史等)")
        elif gap32 > 0.05 and gap64 <= 0.02:
            log("  → 假设B成立: 模型容量不足导致非马尔可夫退化")
            log("    dim=32 下 NM 退化严重, 但 dim=64 下差距消失")
            log("    建议: 使用更大维度, 门控需要足够参数空间")
        elif gap64 < -0.02:
            log("  → 反转! dim=64 下 NM 反超 Vanilla")
            log("    非马尔可夫在高维下有正面贡献")
            log("    建议: 进一步增大 dim 并优化训练策略")
        else:
            log("  → 结论不明确: 差距在噪声范围内")
            log("    建议: 增加实验重复次数或调整超参后重试")

        if v64 > 0.25:
            log(f"\n  Vanilla dim=64 MRR={v64:.4f}, 已接近 TransE(0.294) 水平")
            log("  说明我们的训练流程本身是有效的 (非马尔可夫修改是唯一变量)")
        elif v64 < 0.15:
            log(f"\n  ⚠ 注意: 即使 Vanilla dim=64 MRR={v64:.4f} 仍然很低")
            log("  说明训练流程本身可能有问题 (不只是非马尔可夫修改的问题)")
            log("  建议检查: 子图采样策略、负采样数量、学习率、训练步数")

    log("")


# =================================================================
# 6. 主函数
# =================================================================

def main():
    parser = argparse.ArgumentParser(description="NM-NBFNet 消融实验")
    parser.add_argument('--epochs', type=int, default=15, help='每个配置的最大epoch数')
    parser.add_argument('--steps', type=int, default=5000, help='每epoch最大训练步数')
    parser.add_argument('--configs', nargs='+', default=None,
                        help='指定要跑的配置 (如: vanilla_32 nm_64)')
    args = parser.parse_args()

    log("=" * 70)
    log(" 消融实验: 诊断 NM-NBFNet 性能低下的根因")
    log("=" * 70)
    log(f" 实验设计: 2×2 (Vanilla/NM × dim32/dim64)")
    log(f" 每配置: {args.epochs} epochs, {args.steps} steps/epoch")
    log("")

    train, valid, test, nE, nR = load_data()
    graph = KGGraph(train, nE, nR)
    graph.add_eval_triples(valid, test)

    # 正向 + 反向训练数据
    train_data = []
    for h, r, t in train:
        train_data.append((h, r, t))
        train_data.append((t, r + nR, h))

    configs = get_ablation_configs(gpu_mem_gb, args)

    # 选择要跑的配置
    if args.configs:
        run_configs = {k: v for k, v in configs.items() if k in args.configs}
    else:
        run_configs = configs

    log(f"[Plan] 将运行 {len(run_configs)} 个消融配置:")
    for name, cfg in run_configs.items():
        log(f"  - {cfg['label']} (dim={cfg['dim']}, variant={cfg['variant']})")
    log("")

    all_results = []
    for name, cfg in run_configs.items():
        result = run_single_config(name, cfg, train_data, valid, test, graph, nE, nR)
        all_results.append(result)

    # 分析
    print_diagnosis(all_results)
    analyze_and_plot(all_results)

    # 保存结果
    result_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(result_dir, exist_ok=True)
    save_data = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != 'history'}
        r_copy['val_mrr_history'] = list(zip(r['history']['epochs'], r['history']['val_mrrs']))
        r_copy['loss_history'] = list(zip(r['history']['epochs'], r['history']['losses']))
        save_data.append(r_copy)

    result_file = os.path.join(result_dir, "ablation_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"\n  结果 -> {result_file}")

    log("\n消融实验完成!")
    _log_f.close()


if __name__ == "__main__":
    main()
