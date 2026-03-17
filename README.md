# PCDEN
Non-Markovian extension of NBFNet for knowledge graph reasoning on FB15k-237. Augments Bellman-Ford message passing with path-history awareness. A 2×2 ablation (Vanilla vs NM, dim=32/64) with full-graph training and cross-entropy loss isolates architectural gains from pipeline effects. Evaluated by MRR and Hits@K.
