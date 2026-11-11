"""
Raw data loading, ChemBERTa drug embeddings and drug-graph construction.

Source: Thesis_baselline_undirected_linegraph_GraphSage.ipynb, cells 5, 7, 9, 10,
13, 17 (the notebook that produced ddi_graph_and_linegraph.bin).

Ported as-is. Behaviour that is known to be wrong is kept on purpose and
marked KNOWN ISSUE; it is what produced the thesis graph.
Changes from the notebook (paths/plumbing only):
  * database paths are arguments instead of hard-coded /content/data/... paths;
    create_ddi_graph now passes event_db_path on to get_all_events (the notebook
    called get_all_events() with its default path, which is the same file).
  * create_line_subgraph uses its `original_graph` argument (the notebook read a
    global called `graph`; with num_nodes_to_sample=None, which is how the
    thesis graph was built, both are the same object).
"""

import os
import sqlite3
import zipfile

import pandas as pd
import torch
import dgl


# ---------------------------------------------------------------------------
# raw data download (cell 13)
# ---------------------------------------------------------------------------
def download_raw_data(raw_data_dir: str, gdrive_id: str):
    """Download and extract the META-DDIE databases if they are not there."""
    drug_db = os.path.join(raw_data_dir, "Drug_META_DDIE.db")
    event_db = os.path.join(raw_data_dir, "event_META_DDIE_5foldLabel.db")
    if os.path.exists(drug_db) and os.path.exists(event_db):
        print(f"raw data present in {raw_data_dir}")
        return drug_db, event_db
    import gdown
    parent = os.path.dirname(raw_data_dir.rstrip("/")) or "."
    os.makedirs(parent, exist_ok=True)
    zpath = os.path.join(parent, "data.zip")
    gdown.download(f"https://drive.google.com/uc?export=download&id={gdrive_id}",
                   output=zpath, quiet=False)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(parent)          # the zip contains a top-level data/ folder
    assert os.path.exists(drug_db) and os.path.exists(event_db), (
        f"expected {drug_db} and {event_db} after extraction")
    return drug_db, event_db


# ---------------------------------------------------------------------------
# database loaders (cell 5)
# ---------------------------------------------------------------------------
def load_data_into_dataframe(db_path: str, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_all_events(event_db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(event_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    event_dfs = []
    for table_name in tables:
        df = load_data_into_dataframe(event_db_path, table_name[0])
        event_dfs.append(df)
    return pd.concat(event_dfs)


# ---------------------------------------------------------------------------
# ChemBERTa embeddings (cell 7)
# ---------------------------------------------------------------------------
def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def get_smiles_list(graph, drugs_df, id_to_node):
    smiles = []
    for i in range(graph.num_nodes()):
        drug_id = get_key_by_value(id_to_node, graph.ndata['id'][i].item())
        smiles.append(drugs_df[drugs_df.id == drug_id]['smile'].values[0])
    return smiles


def chunk_and_embed(smiles_string, tokenizer, model, chunk_size=256, overlap=64):
    chunks = [smiles_string[i:i + chunk_size]
              for i in range(0, len(smiles_string), chunk_size - overlap)]
    chunk_embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            chunk_embeddings.append(outputs.last_hidden_state[:, 0, :])  # [CLS] per chunk
    max_len = max(ce.shape[1] for ce in chunk_embeddings)
    padded_embeddings = [torch.nn.functional.pad(ce, (0, max_len - ce.shape[1]))
                         for ce in chunk_embeddings]
    final_embedding = torch.cat(padded_embeddings, dim=0)
    # KNOWN ISSUE (kept): only the first chunk's [CLS] is used; for SMILES
    # longer than 256 characters the remaining chunks are computed and dropped.
    final_embedding = final_embedding[0, :].unsqueeze(0)
    return final_embedding


def generate_smiles_embeddings(graph, drugs_df, id_to_node,
                               model_name="DeepChem/ChemBERTa-77M-MLM"):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    smiles_list = get_smiles_list(graph, drugs_df, id_to_node)
    embeddings = [chunk_and_embed(smile, tokenizer, model) for smile in smiles_list]
    return embeddings


# ---------------------------------------------------------------------------
# drug graph (cells 9, 10)
# ---------------------------------------------------------------------------
def graph_construction(drugs_df, event_df):
    # KNOWN ISSUE (kept): list(set(...)) ordering depends on Python's string
    # hash seed, so node ids differ between Python processes. The thesis graph
    # is reproducible only from the saved ddi_graph_and_linegraph.bin.
    all_ids = list(set(event_df['id1'].unique()).union(set(event_df['id2'].unique())))
    id_to_node = {id_val: idx for idx, id_val in enumerate(all_ids)}

    # Directed: edges only from id1 to id2.
    src_nodes = [id_to_node[id_val] for id_val in event_df['id1']]
    dst_nodes = [id_to_node[id_val] for id_val in event_df['id2']]
    edge_labels = torch.tensor(event_df.label.to_list(), dtype=torch.long)

    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=len(all_ids))
    graph.ndata['id'] = torch.tensor([id_to_node[id_val] for id_val in all_ids])
    graph.ndata['smiles_embedding'] = torch.stack(
        generate_smiles_embeddings(graph, drugs_df, id_to_node), dim=0).squeeze(1)

    graph.edata['label'] = edge_labels
    graph.edata['id'] = torch.arange(graph.num_edges())

    u, v = graph.edges()
    fused_embeddings = (graph.ndata['smiles_embedding'][u] + graph.ndata['smiles_embedding'][v]) / 2
    graph.edata['fused_embedding'] = fused_embeddings
    return graph


def create_ddi_graph(drug_db_path: str, event_db_path: str, neg_sample_ratio: float = 0.3):
    drugs_df = load_data_into_dataframe(drug_db_path, "drug")
    event_df = get_all_events(event_db_path)

    event_df['label'] = event_df['label'].astype(int)
    # events with label >= 228 have 1 event each and are not included, as in META-DDIE
    event_df = event_df[event_df.label <= 227]

    graph = graph_construction(drugs_df, event_df)

    # Negative sampling.
    # KNOWN ISSUES (kept):
    #   * torch.randint is unseeded -> the 333,237 thesis negatives cannot be
    #     regenerated; use the saved .bin to reproduce thesis numbers.
    #   * the rejection test checks only the stored orientation (u, v), so
    #     (v, u) of a recorded DDI can be drawn as "non-interacting".
    #   * no de-duplication among the negatives themselves.
    num_nodes = graph.num_nodes()
    num_positive_edges = graph.num_edges()
    target_negative_samples = int(num_positive_edges * neg_sample_ratio)

    neg_src, neg_dst = [], []
    positive_edges = set(zip(graph.edges()[0].tolist(), graph.edges()[1].tolist()))
    while len(neg_src) < target_negative_samples:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        if src == dst:
            continue
        if (src, dst) not in positive_edges:
            neg_src.append(src)
            neg_dst.append(dst)

    neg_src = torch.tensor(neg_src, dtype=torch.long)
    neg_dst = torch.tensor(neg_dst, dtype=torch.long)
    start_neg_id = graph.num_edges()
    graph.add_edges(neg_src, neg_dst, {
        'label': torch.zeros(neg_src.shape[0], dtype=torch.long),
        'id': torch.arange(start_neg_id, start_neg_id + neg_src.shape[0])})
    return graph


# ---------------------------------------------------------------------------
# line graph (cell 17)
# ---------------------------------------------------------------------------
def create_line_subgraph(original_graph, num_nodes_to_sample=None):
    """Optionally node-subsample the drug graph, then build the directed line
    graph with backtracking=False (an edge (u,v) is adjacent to (v,w), w != u)."""
    if num_nodes_to_sample is None:
        subgraph = original_graph
    else:
        sampled_nodes = torch.randperm(original_graph.num_nodes())[:num_nodes_to_sample]
        subgraph = dgl.node_subgraph(original_graph, sampled_nodes)
    line_subgraph = dgl.line_graph(subgraph, backtracking=False)
    print("Original subgraph:", subgraph)
    print("Created line subgraph:", line_subgraph)
    return subgraph, line_subgraph


def save_graph_pair(path, subgraph, line_subgraph):
    """Same file layout as the thesis .bin: [drug graph, line graph]."""
    dgl.save_graphs(path, [subgraph, line_subgraph])
