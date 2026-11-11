import os
import sqlite3
import zipfile

import pandas as pd
import torch
import dgl


def download_raw_data(drug_db: str, event_db: str, gdrive_id: str):
    """Download and extract the META-DDIE databases if they are not there.
    drug_db / event_db come from ddi.paths (paths.drug_db, paths.event_db);
    both must sit in the same data/ folder, as in the zip."""
    if os.path.exists(drug_db) and os.path.exists(event_db):
        print(f"raw data present: {drug_db}, {event_db}")
        return drug_db, event_db
    raw_data_dir = os.path.dirname(drug_db)
    assert os.path.dirname(event_db) == raw_data_dir, "both databases must be in the same folder"
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

def get_smiles_list(graph, drugs_df, id_to_node):
    node_to_id = {node: drug_id for drug_id, node in id_to_node.items()}
    smiles = []
    for i in range(graph.num_nodes()):
        drug_id = node_to_id[graph.ndata['id'][i].item()]
        smiles.append(drugs_df[drugs_df.id == drug_id]['smile'].values[0])
    return smiles

def chunk_and_embed(smiles_string, tokenizer, model, chunk_size=256, overlap=64):
    """ChemBERTa embedding of one SMILES: the mean of the [CLS] vectors of
    overlapping 256-character chunks. A SMILES of <= 256 characters is a single
    chunk, so its embedding is unchanged from the thesis.

    The thesis graph (ddi_graph_and_linegraph.bin) used only the FIRST chunk's
    [CLS]; that affected 31 of 3,535 drugs (0.97% of interactions)."""
    # Chunk starts 0, 192, 384, ... A new chunk starts only while the previous one
    # has not reached the end of the string, so no chunk lies entirely inside the
    # previous one and any SMILES of <= 256 characters is exactly one chunk.
    step = chunk_size - overlap
    starts = range(0, max(len(smiles_string) - overlap, 1), step)
    chunks = [smiles_string[i:i + chunk_size] for i in starts]
    cls = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt")
        with torch.no_grad():
            cls.append(model(**inputs).last_hidden_state[:, 0, :])   # [1, hidden]
    return torch.cat(cls, dim=0).mean(dim=0, keepdim=True)            # [1, hidden]


def generate_smiles_embeddings(graph, drugs_df, id_to_node,
                               model_name="DeepChem/ChemBERTa-77M-MLM"):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    smiles_list = get_smiles_list(graph, drugs_df, id_to_node)
    embeddings = [chunk_and_embed(smile, tokenizer, model) for smile in smiles_list]
    return embeddings

def graph_construction(drugs_df, event_df):
    # Drugs are numbered in sorted DrugBank-id order, so every build gives the
    # same node ids. The thesis graph (ddi_graph_and_linegraph.bin) was built
    # with list(set(...)), whose order depended on Python's string-hash seed;
    # reproduce thesis numbers only from that saved file.
    all_ids = sorted(set(event_df['id1'].unique()).union(set(event_df['id2'].unique())))
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
