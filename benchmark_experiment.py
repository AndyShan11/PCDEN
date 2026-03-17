"""
NM-NBFNet vs Vanilla NBFNet -- FB15k-237 公平对比实验 (修复版v3)
================================================================
关键修复 (基于消融实验诊断):
  1. 全图训练 — 消除训练/评估分布不匹配 (原版用子图训练、全图评估)
  2. 全实体打分 + CrossEntropy — 替代自对抗负采样, 梯度信号更强
  3. 充分训练 — 每epoch覆盖更多数据, 更多epoch
  4. 两种变体公平对比 — 相同训练流程, 只改BellmanFord层

消融实验结论:
  Vanilla dim=32 MRR=0.081, NM dim=32 MRR=0.097
  → NM修改不是问题, 训练管线才是瓶颈

用法:
  python benchmark_experiment.py                      # 默认: 两种变体, dim=32
  python benchmark_experiment.py --dim 64             # dim=64
  python benchmark_experiment.py --variant vanilla    # 只跑vanilla
  python benchmark_experiment.py --steps 20000        # 更多步数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, time, random, math, argparse, json
import urllib.request
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "benchmark_log.txt")
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

DATA_DIR = os.path.join(BASE_DIR, "data", "FB15k-237")


# =================================================================
# 1. 数据集
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


# =================================================================
# 2. 图结构 (全图预加载到GPU)
# =================================================================

class KGGraph:
    def __init__(self, triples, nE, nR):
        self.nE, self.nR = nE, nR
        self.total_rel = nR * 2

        # 构建完整边列表 (含反向边)
        src, tgt, rel = [], [], []
        for h, r, t in triples:
            src.append(h); tgt.append(t); rel.append(r)
            src.append(t); tgt.append(h); rel.append(r + nR)
        self.full_src = torch.tensor(src, dtype=torch.long)
        self.full_tgt = torch.tensor(tgt, dtype=torch.long)
        self.full_rel = torch.tensor(rel, dtype=torch.long)
        self.num_edges = len(src)

        # 预加载到GPU (全图训练的关键)
        self.gpu_src = self.full_src.to(device)
        self.gpu_tgt = self.full_tgt.to(device)
        self.gpu_rel = self.full_rel.to(device)

        # true_tails 用于 filtered 评估
        self.true_tails = defaultdict(lambda: defaultdict(set))
        for h, r, t in triples:
            self.true_tails[h][r].add(t)
            self.true_tails[t][r + nR].add(h)

        log(f"[Graph] 边数(含反向): {self.num_edges}, 平均度: {self.num_edges/nE:.1f}")

    def add_eval_triples(self, valid_triples, test_triples):
        for triples in [valid_triples, test_triples]:
            for h, r, t in triples:
                self.true_tails[h][r].add(t)
                self.true_tails[t][r + self.nR].add(h)


# =================================================================
# 3. 模型: Vanilla NBFNet vs NM-NBFNet
# =================================================================

class VanillaBellmanFordLayer(nn.Module):
    """标准 Bellman-Ford 消息传递 (对照组)"""
    def __init__(self, hidden_dim, num_relations):
        super().__init__()
        self.msg_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_src, edge_tgt, edge_rel, nE):
        batch_size, _, dim = h.shape
        r = self.rel_emb(edge_rel)
        h_src = h[:, edge_src]
        msg = self.msg_linear(h_src) * r.unsqueeze(0)
        idx = edge_tgt.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, dim)
        h_agg = torch.zeros(batch_size, nE, dim, device=h.device)
        h_agg.scatter_add_(1, idx, msg)
        h_new = self.norm(h + F.relu(self.update(torch.cat([h, h_agg], dim=-1))))
        return h_new


class NMBellmanFordLayer(nn.Module):
    """非马尔可夫 Bellman-Ford — 历史感知门控 (实验组)"""
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
        gate = torch.sigmoid(self.gate_w(h_src))  # 非马尔可夫门控
        msg = self.msg_linear(h_src) * r.unsqueeze(0) * gate
        idx = edge_tgt.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, dim)
        h_agg = torch.zeros(batch_size, nE, dim, device=h.device)
        h_agg.scatter_add_(1, idx, msg)
        h_new = self.norm(h + F.relu(self.update(torch.cat([h, h_agg], dim=-1))))
        return h_new


class NBFNet(nn.Module):
    """统一 NBFNet — variant='vanilla' 或 'nm'"""

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
            # 负采样模式 (用于兼容, 本版本不使用)
            K = targets.shape[1]
            bi = torch.arange(bs, device=h.device).unsqueeze(1).expand(-1, K)
            ht = h[bi, targets]
            te = self.ent_emb(targets)
            return self.score_mlp(torch.cat([ht, te], -1)).squeeze(-1)
        else:
            # 全实体打分 (本版本的核心改进)
            te = self.ent_emb.weight.unsqueeze(0).expand(bs, -1, -1)
            return self.score_mlp(torch.cat([h, te], -1)).squeeze(-1)


# =================================================================
# 4. 训练 — 全实体打分 + CrossEntropy
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


def train_epoch(model, train_data, graph, optimizer, nE,
                batch_size=4, max_steps=10000, scaler=None,
                accum_steps=4, scheduler=None):
    """
    全实体打分 + CrossEntropy 训练.
    关键改进: 不再使用负采样, 直接对全部14541个实体打分.
    """
    model.train()
    random.shuffle(train_data)
    total_loss, n = 0, 0
    use_amp = (device.type == "cuda" and scaler is not None)

    # 使用预加载到GPU的全图 (不再采样子图!)
    e_src, e_tgt, e_rel = graph.gpu_src, graph.gpu_tgt, graph.gpu_rel

    optimizer.zero_grad()
    step_in_accum = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    total_iters = min(len(train_data), max_steps * batch_size)
    for i in range(0, total_iters, batch_size):
        batch = train_data[i:i + batch_size]
        if not batch:
            continue

        heads = torch.tensor([x[0] for x in batch], device=device)
        rels  = torch.tensor([x[1] for x in batch], device=device)
        tails = torch.tensor([x[2] for x in batch], device=device)

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                scores = model(heads, rels, e_src, e_tgt, e_rel)  # [B, nE] 全实体打分
                loss = F.cross_entropy(scores, tails) / accum_steps
            scaler.scale(loss).backward()
        else:
            scores = model(heads, rels, e_src, e_tgt, e_rel)
            loss = F.cross_entropy(scores, tails) / accum_steps
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

    gpu_mem = 0
    if device.type == "cuda":
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**2

    return total_loss / max(n, 1), gpu_mem


# =================================================================
# 5. 评估 — Filtered ranking
# =================================================================

@torch.no_grad()
def evaluate(model, triples, graph, nE, max_test=1000, batch_eval=4):
    model.eval()
    subset = triples[:max_test]
    ranks = []

    e_src, e_tgt, e_rel = graph.gpu_src, graph.gpu_tgt, graph.gpu_rel

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
    return mrr, h1, h3, h10, ranks


# =================================================================
# 6. 单变体训练流程
# =================================================================

def run_variant(variant, train_data, valid, test, graph, nE, nR, args):
    """完整训练一个变体, 返回结果字典"""
    label = "NM-NBFNet" if variant == "nm" else "Vanilla-NBFNet"
    log(f"\n{'='*65}")
    log(f" {label} (dim={args.dim}, layers={args.layers})")
    log(f"{'='*65}")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = NBFNet(nE, nR, dim=args.dim, num_layers=args.layers,
                   dropout=args.dropout, variant=variant).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    log(f"  参数量: {nparams:,}")
    log(f"  variant={variant}, batch={args.batch}x{args.accum}(accum)={args.batch*args.accum}")
    log(f"  loss=CrossEntropy(全实体打分), lr={args.lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    steps_per_epoch_optim = args.steps // args.accum
    total_training_steps = steps_per_epoch_optim * args.epochs
    warmup_steps = steps_per_epoch_optim * 2  # 2 epoch warmup
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_training_steps)

    best_mrr = 0
    patience_counter = 0
    save_path = os.path.join(DATA_DIR, f"best_{variant}.pt")

    history = {
        'epochs': [], 'losses': [], 'val_mrrs': [], 'val_h1s': [],
        'val_h3s': [], 'val_h10s': [], 'lrs': [], 'gpu_mems': [],
    }
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        te = time.time()

        loss, gpu_mem = train_epoch(
            model, train_data, graph, optimizer, nE,
            batch_size=args.batch, max_steps=args.steps,
            scaler=scaler, accum_steps=args.accum, scheduler=scheduler,
        )
        ep_time = time.time() - te
        current_lr = optimizer.param_groups[0]['lr']

        mrr, h1, h3, h10, _ = evaluate(model, valid, graph, nE,
                                         max_test=args.val_samples, batch_eval=4)

        history['epochs'].append(epoch)
        history['losses'].append(loss)
        history['val_mrrs'].append(mrr)
        history['val_h1s'].append(h1)
        history['val_h3s'].append(h3)
        history['val_h10s'].append(h10)
        history['lrs'].append(current_lr)
        history['gpu_mems'].append(gpu_mem)

        marker = ""
        if mrr > best_mrr:
            best_mrr = mrr
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            marker = " *"
        else:
            patience_counter += 1

        log(f"  Epoch {epoch:2d} | loss={loss:.4f} | MRR={mrr:.4f} H@1={h1:.1f} H@3={h3:.1f} H@10={h10:.1f} "
            f"| lr={current_lr:.1e} | {ep_time:.0f}s"
            + (f" | GPU={gpu_mem/1024:.1f}GB" if gpu_mem > 0 else "")
            + marker)

        if patience_counter >= args.patience:
            log(f"  Early stopping (patience={args.patience})")
            break

    train_time = (time.time() - t0) / 60
    total_epochs = len(history['losses'])

    # 测试
    log(f"  加载最佳模型 (Best Val MRR={best_mrr:.4f})...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    test_mrr, test_h1, test_h3, test_h10, test_ranks = evaluate(
        model, test, graph, nE, max_test=args.test_samples, batch_eval=4
    )
    log(f"  Test: MRR={test_mrr:.4f} H@1={test_h1:.1f}% H@3={test_h3:.1f}% H@10={test_h10:.1f}%")
    log(f"  训练: {total_epochs} epochs, {train_time:.1f}分钟")

    # 清理
    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        'variant': variant, 'label': label, 'nparams': nparams,
        'dim': args.dim, 'train_time_min': train_time, 'total_epochs': total_epochs,
        'best_val_mrr': best_mrr,
        'test_mrr': test_mrr, 'test_h1': test_h1, 'test_h3': test_h3, 'test_h10': test_h10,
        'test_ranks': test_ranks, 'history': history,
    }


# =================================================================
# 7. 对比分析 + 可视化
# =================================================================

def compare_and_plot(results, args):
    """生成对比分析和图表"""

    # 结果对比表
    log("\n" + "=" * 70)
    log(" FB15k-237 测试集 (Filtered) — 对比结果")
    log("=" * 70)

    log(f"\n{'Method':<38} {'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7}")
    log("-" * 70)

    # 文献参考值
    refs = [
        ("TransE (2013)",               .294, .197, .337, .497),
        ("DistMult (2015)",             .241, .155, .263, .419),
        ("ComplEx (2016)",              .247, .158, .275, .428),
        ("ConvE (2018)",                .325, .237, .356, .501),
        ("RotatE (2019)",               .338, .241, .375, .533),
        ("TuckER (2019)",               .358, .266, .394, .544),
        ("CompGCN (2020)",              .355, .264, .390, .535),
        ("NBFNet (2021)",               .415, .321, .454, .599),
    ]
    for name, mrr, h1, h3, h10 in refs:
        log(f"  {name:<36} {mrr:>7.3f} {h1:>7.3f} {h3:>7.3f} {h10:>7.3f}")

    log("-" * 70)
    for r in results:
        name = f"{r['label']} ({r['train_time_min']:.0f}min, {r['total_epochs']}ep)"
        log(f"  {name:<36} {r['test_mrr']:>7.3f} {r['test_h1']/100:>7.3f} "
            f"{r['test_h3']/100:>7.3f} {r['test_h10']/100:>7.3f}  <--")

    # NM vs Vanilla 对比
    if len(results) == 2:
        v = next((r for r in results if r['variant'] == 'vanilla'), None)
        nm = next((r for r in results if r['variant'] == 'nm'), None)
        if v and nm:
            log(f"\n  [对比分析]")
            d_mrr = nm['test_mrr'] - v['test_mrr']
            d_h1 = nm['test_h1'] - v['test_h1']
            d_h10 = nm['test_h10'] - v['test_h10']
            log(f"  NM vs Vanilla MRR 差: {d_mrr:+.4f}")
            log(f"  NM vs Vanilla H@1 差: {d_h1:+.1f}%")
            log(f"  NM vs Vanilla H@10差: {d_h10:+.1f}%")
            if d_mrr > 0.005:
                log(f"  结论: NM-NBFNet 优于 Vanilla (MRR +{d_mrr:.4f})")
            elif d_mrr < -0.005:
                log(f"  结论: Vanilla 优于 NM-NBFNet (MRR {d_mrr:.4f})")
            else:
                log(f"  结论: 两者性能相当 (差距 {abs(d_mrr):.4f} < 0.005)")

    # 绘图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matplotlib.rcParams['font.size'] = 11
    except ImportError:
        log("[Plot] matplotlib 未安装, 跳过")
        return

    fig_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    colors = {'vanilla': '#1f77b4', 'nm': '#d62728'}
    markers = {'vanilla': 'o', 'nm': 's'}

    # ---- 图1: 训练曲线 (Loss + MRR) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        c, m = colors[r['variant']], markers[r['variant']]
        ax1.plot(r['history']['epochs'], r['history']['losses'],
                 f'-{m}', color=c, lw=2, ms=5, label=r['label'])
        ax2.plot(r['history']['epochs'], r['history']['val_mrrs'],
                 f'-{m}', color=c, lw=2, ms=5, label=r['label'])

    for name, val, ls in [("NBFNet", 0.415, '--'), ("TransE", 0.294, ':')]:
        ax2.axhline(y=val, color='gray', ls=ls, alpha=0.5, label=f"{name} ({val})")

    ax1.set(xlabel="Epoch", ylabel="CrossEntropy Loss", title="Training Loss")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax2.set(xlabel="Epoch", ylabel="MRR", title="Validation MRR")
    ax2.set_ylim(bottom=0); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"NM-NBFNet vs Vanilla — dim={args.dim}", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "comparison_curves.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  [Plot] -> figures/comparison_curves.png")

    # ---- 图2: 测试集柱状对比 ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    metrics = [
        ('test_mrr', 'MRR', False),
        ('test_h1', 'Hits@1 (%)', True),
        ('test_h3', 'Hits@3 (%)', True),
        ('test_h10', 'Hits@10 (%)', True),
    ]
    for ax, (key, title, is_pct) in zip(axes, metrics):
        labels, vals, bar_colors = [], [], []
        for r in results:
            labels.append(r['label'])
            vals.append(r[key])
            bar_colors.append(colors[r['variant']])
        bars = ax.bar(range(len(labels)), vals, color=bar_colors, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.set_title(title); ax.grid(axis='y', alpha=0.3)
        for idx, v in enumerate(vals):
            fmt = f'{v:.4f}' if not is_pct else f'{v:.1f}'
            ax.text(idx, v + max(vals)*0.02, fmt, ha='center', fontsize=10, fontweight='bold')
    fig.suptitle(f"Test Performance Comparison (dim={args.dim})", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "comparison_bars.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  [Plot] -> figures/comparison_bars.png")

    # ---- 图3: SOTA 方法对比柱状图 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    methods = ['TransE', 'DistMult', 'ComplEx', 'ConvE', 'RotatE', 'TuckER', 'CompGCN', 'NBFNet']
    mrrs   = [.294, .241, .247, .325, .338, .358, .355, .415]
    h1s    = [.197, .155, .158, .237, .241, .266, .264, .321]
    h3s    = [.337, .263, .275, .356, .375, .394, .390, .454]
    h10s   = [.497, .419, .428, .501, .533, .544, .535, .599]

    for r in results:
        methods.append(r['label'])
        mrrs.append(r['test_mrr'])
        h1s.append(r['test_h1'] / 100)
        h3s.append(r['test_h3'] / 100)
        h10s.append(r['test_h10'] / 100)

    for ax, vals, metric_name, color in [
        (axes[0,0], mrrs,  'MRR',     '#2196F3'),
        (axes[0,1], h1s,   'Hits@1',  '#4CAF50'),
        (axes[1,0], h3s,   'Hits@3',  '#FF9800'),
        (axes[1,1], h10s,  'Hits@10', '#E91E63'),
    ]:
        bar_colors = [color] * (len(methods) - len(results))
        for r in results:
            bar_colors.append(colors[r['variant']])
        alphas = [0.5] * (len(methods) - len(results)) + [1.0] * len(results)
        bars = ax.bar(range(len(methods)), vals, color=bar_colors, edgecolor='white')
        for b, a in zip(bars, alphas):
            b.set_alpha(a)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=8)
        ax.set_ylabel(metric_name); ax.set_title(metric_name)
        ax.grid(axis='y', alpha=0.3)
        for idx, v in enumerate(vals):
            ax.text(idx, v + 0.005, f'{v:.3f}', ha='center', fontsize=7,
                    fontweight='bold' if idx >= len(methods)-len(results) else 'normal')

    fig.suptitle(f"FB15k-237 Benchmark Comparison", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "sota_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  [Plot] -> figures/sota_comparison.png")

    # ---- 图4: Rank 分布 (如果有两个变体) ----
    if len(results) >= 2 and all('test_ranks' in r for r in results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for r in results:
            ranks = r['test_ranks']
            ranks_clip = ranks[ranks <= 100]
            ax1.hist(ranks_clip, bins=50, alpha=0.6, color=colors[r['variant']],
                     label=f"{r['label']} (MRR={r['test_mrr']:.4f})", edgecolor='white')
            ax2.hist(ranks, bins=np.logspace(0, np.log10(max(ranks.max(), 2)), 50),
                     alpha=0.6, color=colors[r['variant']],
                     label=f"{r['label']} (med={np.median(ranks):.0f})", edgecolor='white')
        ax1.set(xlabel="Rank", ylabel="Count", title="Rank Distribution (rank <= 100)")
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set(xlabel="Rank (log)", ylabel="Count", title="Full Rank Distribution")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        fig.suptitle("Test Set Rank Distribution", fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "rank_distribution.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        log(f"  [Plot] -> figures/rank_distribution.png")

    log(f"  所有图表 -> figures/")


# =================================================================
# 8. 保存结果
# =================================================================

def save_results(results, args):
    result_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(result_dir, exist_ok=True)

    save_data = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k not in ('test_ranks', 'history')}
        r_copy['val_mrr_history'] = list(zip(r['history']['epochs'], r['history']['val_mrrs']))
        r_copy['loss_history'] = list(zip(r['history']['epochs'], r['history']['losses']))
        save_data.append(r_copy)

    result_file = os.path.join(result_dir, "benchmark_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"\n  结果 -> {result_file}")

    # 文本版结果
    txt_file = os.path.join(result_dir, "benchmark_results.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"实验配置: dim={args.dim}, layers={args.layers}, lr={args.lr}\n")
        f.write(f"训练: {args.steps} steps/epoch, batch={args.batch}x{args.accum}, loss=CrossEntropy\n")
        f.write(f"改进: 全图训练 + 全实体打分 + CE loss\n\n")
        for r in results:
            f.write(f"{r['label']}: MRR={r['test_mrr']:.4f} H@1={r['test_h1']:.1f}% "
                    f"H@3={r['test_h3']:.1f}% H@10={r['test_h10']:.1f}% "
                    f"({r['train_time_min']:.1f}min, {r['total_epochs']}ep)\n")
    log(f"  结果 -> {txt_file}")


# =================================================================
# 9. 主函数
# =================================================================

def main():
    parser = argparse.ArgumentParser(description="NM-NBFNet vs Vanilla NBFNet 对比实验")
    parser.add_argument('--dim', type=int, default=32, help='Embedding维度')
    parser.add_argument('--layers', type=int, default=3, help='BellmanFord层数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--accum', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--epochs', type=int, default=20, help='最大epoch数')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--steps', type=int, default=10000, help='每epoch最大步数')
    parser.add_argument('--val_samples', type=int, default=1000, help='验证集采样数')
    parser.add_argument('--test_samples', type=int, default=5000, help='测试集采样数')
    parser.add_argument('--variant', type=str, default='both',
                        choices=['both', 'vanilla', 'nm'], help='运行哪个变体')
    args = parser.parse_args()

    log("=" * 70)
    log(" NM-NBFNet vs Vanilla NBFNet — FB15k-237 (修复版v3)")
    log("=" * 70)
    log(f" 关键改进: 全图训练 + 全实体CE打分 (替代子图采样+负采样)")
    log(f" 配置: dim={args.dim}, layers={args.layers}, lr={args.lr}")
    log(f" 训练: batch={args.batch}x{args.accum}={args.batch*args.accum}, "
        f"{args.steps} steps/epoch, max {args.epochs} epochs")
    log("")

    train, valid, test, nE, nR = load_data()
    graph = KGGraph(train, nE, nR)
    graph.add_eval_triples(valid, test)

    # 正向 + 反向训练数据
    train_data = []
    for h, r, t in train:
        train_data.append((h, r, t))
        train_data.append((t, r + nR, h))
    log(f"[Train] 训练样本: {len(train_data)} (含反向)")
    log(f"[Train] 每epoch步数: {args.steps}, 覆盖样本: {args.steps*args.batch}/{len(train_data)} "
        f"({args.steps*args.batch/len(train_data)*100:.0f}%)")

    # 选择变体
    variants = []
    if args.variant in ('both', 'vanilla'):
        variants.append('vanilla')
    if args.variant in ('both', 'nm'):
        variants.append('nm')

    log(f"[Plan] 将运行: {', '.join(variants)}")

    results = []
    for v in variants:
        r = run_variant(v, train_data, valid, test, graph, nE, nR, args)
        results.append(r)

    compare_and_plot(results, args)
    save_results(results, args)

    log("\n实验完成!")
    _log_f.close()


if __name__ == "__main__":
    main()
