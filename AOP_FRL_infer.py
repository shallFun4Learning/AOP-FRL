import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import torch
from transformers import TrainingArguments, Trainer
import numpy as np
from torch.nn import functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
from torch import nn
import Getmetrics
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
import datasets

k=int(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
data_file_path = "/data/xuxf/antiO_p70/data/03_p70_AO_db_smiles_p.csv"
esm2_model_path = '/data/xuxf/antiO_p70/esm2_t30_150M_UR50D'
model_file_path=r'/data/xuxf/ANTIO241214/ANTIO/esm+gcn-6-a+r-ag2e-4(777？)'
res_path=f'/data/xuxf/antiO_p70/final{k}_metrics_inf.csv'
def atom_features(atom):
    features = np.array([
        atom.GetAtomicNum(),
        atom.GetTotalNumHs(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetImplicitValence(),
        atom.GetNumRadicalElectrons(),
        atom.GetTotalValence(),
        atom.IsInRing()
    ], dtype=np.float32)
    return features


def bond_features(bond):
    features = np.array([
        bond.GetBondTypeAsDouble(),
        bond.GetIsAromatic(),
        bond.IsInRing(),
        bond.GetBondDir()
    ], dtype=np.float32)
    return features

def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)
        
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        
        atom_features_list = [atom_features(atom) for atom in atoms]
        num_nodes = len(atoms)

        edge_index = []
        edge_attr = []
        for bond in bonds:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
            bond_feature = bond_features(bond)
            edge_attr.append(bond_feature)
            edge_attr.append(bond_feature)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)


        new_x = torch.zeros((num_nodes, len(atom_features_list[0]) + edge_attr.size(1)))


        for i, atom_feature in enumerate(atom_features_list):
            connected_edges = edge_index[1] == i
            if connected_edges.any():

                connected_edge_features = edge_attr[connected_edges].mean(dim=0)
                concatenated_features = torch.cat((torch.tensor(atom_feature, dtype=torch.float), connected_edge_features))
                new_x[i] = concatenated_features
            else:
                new_x[i][:len(atom_feature)] = torch.tensor(atom_feature, dtype=torch.float)  # 仅填充节点特征部分
        
        return Data(x=new_x, edge_index=edge_index, edge_attr=edge_attr)
    
    except ValueError as e:
        print(e)
        raise SystemExit("Invalid data detected, stopping execution.")

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.N = 2
        self.conv1 = GCNConv(14, 128 * self.N)
        self.bn1 = BatchNorm(128 * self.N)
        self.conv2 = GCNConv(128 * self.N, 256 * self.N)
        self.bn2 = BatchNorm(256 * self.N)
        self.conv3 = GCNConv(256 * self.N, 256 * self.N)
        self.bn3 = BatchNorm(256 * self.N)
        
        self.fc1 = nn.Linear(256 * self.N, 640)
        self.dropout = nn.Dropout(p=0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = self.conv2(x, edge_index)
        x = F.relu(self.bn2(x))
        x = self.conv3(x, edge_index)
        x = F.relu(self.bn3(x))
        
        x = global_mean_pool(x, batch)
        
        x_fc = self.dropout(F.relu(self.fc1(x)))
        return x_fc

class CombinedModel(nn.Module):
    def __init__(self, esm2_model_name):
        super(CombinedModel, self).__init__()
        self.esm2_model = AutoModel.from_pretrained(esm2_model_name, ignore_mismatched_sizes=True)
        for param in self.esm2_model.parameters():
            param.requires_grad = False
        self.gcn_model = GCN()
        self.esm2_hidden_size = self.esm2_model.config.hidden_size
        self.sequence_length = 64

        self.conv1d = nn.Conv1d(in_channels=self.sequence_length, out_channels=1, kernel_size=1)

        self.attention = nn.MultiheadAttention(embed_dim=self.esm2_hidden_size * 2, num_heads=8)
        
        self.fc1 = nn.Linear(self.esm2_hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)

        self.classifier = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0)

    def forward(self, esm2_input_ids=None, esm2_attention_mask=None, graph=None, labels=None):
        esm2_outputs = self.esm2_model(
            esm2_input_ids,
            attention_mask=esm2_attention_mask
        )
        esm2_output = esm2_outputs.last_hidden_state

        esm2_output_conv = self.conv1d(esm2_output)
        esm2_output_conv = esm2_output_conv.squeeze(1)

        gcn_output = self.gcn_model(graph)

        combined_features = torch.cat((esm2_output_conv, gcn_output), dim=1)
        combined_features = combined_features.unsqueeze(1)

        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
        attn_output = attn_output.squeeze(1)

        x = combined_features.squeeze(1) + attn_output

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.classifier(self.dropout(x))
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, 2), labels.view(-1))
            outputs = (loss,) + (logits,)
            return outputs
        else:
            return logits

class CombinedDataset(Dataset):
    def __init__(self, csv_file, esm2_model_name, max_length=64):
        self.protein_data = pd.read_csv(csv_file)
        
        self.esm2_tokenizer = AutoTokenizer.from_pretrained(esm2_model_name)
        
        self.max_length = max_length

        self.graph_data = datasets.load_dataset("csv", cache_dir='/data/xuxf/antiO_p70/_cache',
                                                data_files=csv_file)
        self.graph_data = self.graph_data.map(lambda x: {"SMILES": x["SMILES"], "label": x["label"]})

    def __len__(self):
        return len(self.protein_data)

    def __getitem__(self, idx):
        sequence = self.protein_data.iloc[idx]['sequence']
        label = self.protein_data.iloc[idx]['label']

        esm2_encoded = self.esm2_tokenizer(sequence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        smiles = self.graph_data['train'][int(idx)]["SMILES"]

        graph = smiles_to_graph(smiles)
        
        item = {
            'sequence': sequence,
            'esm2_input_ids': esm2_encoded['input_ids'].squeeze(0),
            'esm2_attention_mask': esm2_encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'graph': graph
        }

        return item

def custom_collate_fn(batch):
    esm2_input_ids = torch.stack([item['esm2_input_ids'] for item in batch])
    esm2_attention_mask = torch.stack([item['esm2_attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    graphs = Batch.from_data_list([item['graph'] for item in batch])
    return {
        'esm2_input_ids': esm2_input_ids,
        'esm2_attention_mask': esm2_attention_mask,
        'labels': labels,
        'graph': graphs
    }


mydataset = CombinedDataset(csv_file=data_file_path, esm2_model_name=esm2_model_path)
df = pd.read_csv(data_file_path)
labels = [data['labels'].item() for data in mydataset]  
all_fold_metrics = []

for fold in range(5):
    print(f"Inference for fold {fold + 1}")
    torch.cuda.empty_cache()
    import os

    # Check paths for gn=0 to gn=10
    for gn in range(11):
        outPutDir = model_file_path+f'/output{gn}_fold{fold + 1}'
        #GN refers to GPU number during training
        #When training with multiple GPUs, 
        # it is recommended to manually specify outPutDir
        if os.path.exists(outPutDir):
            break

    # outPutDir = model_file_path+f'/output{k}_fold{fold + 1}'
  
    checkpoint_dirs = [d for d in os.listdir(outPutDir) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(outPutDir, d))]
    if not checkpoint_dirs:
        raise Exception(f"No checkpoints found in {outPutDir}")
    last_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
    last_checkpoint_path = os.path.join(outPutDir, last_checkpoint)
    

    mymodel = CombinedModel(esm2_model_name=esm2_model_path)
    mymodel = mymodel.cuda()
    

    model_state_dict = torch.load(os.path.join(last_checkpoint_path, 'pytorch_model.bin'))
    mymodel.load_state_dict(model_state_dict)
    

    test_dataset = CombinedDataset(csv_file=data_file_path, esm2_model_name=esm2_model_path)
    

    test_idx = df[df['partition'] == fold].index.tolist()
    test_subset = torch.utils.data.Subset(test_dataset, test_idx)
    

    training_args = TrainingArguments(
        output_dir=str(outPutDir),
        per_device_eval_batch_size=32,
    )
    

    trainer = Trainer(
        model=mymodel,
        args=training_args,
        data_collator=custom_collate_fn, 
    )
    

    predictions = trainer.predict(test_subset)
    metrics = Getmetrics.getMetrics((predictions.predictions, predictions.label_ids))
    all_fold_metrics.append(metrics)
    
    logits_numpy = np.array(predictions.predictions)
    
    res_dict = {
        'sequence': [mydataset[i]['sequence'] for i in test_idx],
        'score': [Getmetrics.getScore(logits_numpy[i]) for i in range(len(logits_numpy))],
        'predict_label': [Getmetrics.getPredictLabel(logits_numpy[i]) for i in range(len(logits_numpy))],
        'y_label': predictions.label_ids,
    }
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv(outPutDir + os.sep + 'predict_res_inf.csv', index=None)
    
    res_met_df = pd.DataFrame(metrics, index=[1]).T
    res_met_df.to_csv(outPutDir + os.sep + 'metrics_res_inf.csv')

mean_metrics = {metric: np.mean([fold_metrics[metric] for fold_metrics in all_fold_metrics]) for metric in all_fold_metrics[0]}

print("Mean metrics:", mean_metrics)

all_fold_metrics_df = pd.DataFrame(all_fold_metrics)
mean_metrics_df = pd.DataFrame([mean_metrics], index=['mean'])

final_metrics_df = pd.concat([all_fold_metrics_df, mean_metrics_df])
final_metrics_df.to_csv()
