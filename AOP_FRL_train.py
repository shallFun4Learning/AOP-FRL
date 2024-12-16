import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
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
data_file_path = "/data/xuxf/ANTIO/data/03_p90_AO_db_smiles_p.csv"
esm2_model_path = '/data/xuxf/antiO_p70/esm2_t30_150M_UR50D'
out_file_path='/data/xuxf/ANTIO/outfile/mymodel'
res_path=f'/data/xuxf/ANTIO/final{k}_metrics_inf.csv'
# 定义原子特征的提取函数
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

# 定义化学键特征的提取函数
def bond_features(bond):
    features = np.array([
        bond.GetBondTypeAsDouble(),
        bond.GetIsAromatic(),
        bond.IsInRing(),
        bond.GetBondDir()
    ], dtype=np.float32)
    return features

# 转换SMILES到分子对象的函数
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

        # 确保new_x的大小与所有特征的大小匹配
        new_x = torch.zeros((num_nodes, len(atom_features_list[0]) + edge_attr.size(1)))

        # 将边特征与节点特征拼接
        for i, atom_feature in enumerate(atom_features_list):
            connected_edges = edge_index[1] == i
            if connected_edges.any():
                # 对所有连接的边特征求平均并与节点特征拼接
                connected_edge_features = edge_attr[connected_edges].mean(dim=0)
                concatenated_features = torch.cat((torch.tensor(atom_feature, dtype=torch.float), connected_edge_features))
                new_x[i] = concatenated_features
            else:
                new_x[i][:len(atom_feature)] = torch.tensor(atom_feature, dtype=torch.float)  # 仅填充节点特征部分
        
        return Data(x=new_x, edge_index=edge_index, edge_attr=edge_attr)
    
    except ValueError as e:
        print(e)
        raise SystemExit("Invalid data detected, stopping execution.")

# 数据增强函数
def augment_sequence(sequence, prob=0.1):
    """随机替换、插入或删除肽段序列中的氨基酸"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 常见的20种氨基酸
    new_sequence = list(sequence)

    i = 0
    while i < len(new_sequence):
        if random.random() < prob:
            choice = random.choice(['replace', 'insert', 'delete'])
            if choice == 'replace':
                new_sequence[i] = random.choice(amino_acids)
            elif choice == 'insert':
                new_sequence.insert(i, random.choice(amino_acids))
                i += 1  # 插入后跳过新插入的元素
            elif choice == 'delete' and len(new_sequence) > 1:
                new_sequence.pop(i)
                continue  # 删除后跳过索引增加以防止越界
        i += 1

    return ''.join(new_sequence)


def reverse_sequence(sequence):
    return sequence[::-1]

def augment_smiles(smiles, num_augmentations=5):
    """通过随机断键和重排增强SMILES序列"""
    mol = Chem.MolFromSmiles(smiles)
    augmented_smiles = set()
    for _ in range(num_augmentations):
        Chem.Kekulize(mol, clearAromaticFlags=True)
        rwmol = Chem.RWMol(mol)
        num_bonds = rwmol.GetNumBonds()

        if num_bonds > 0:
            # 随机选择一个化学键的索引
            bond_idx = random.randint(0, num_bonds - 1)
            bond = rwmol.GetBondWithIdx(bond_idx)

            # 获取两个原子索引
            atom1 = bond.GetBeginAtomIdx()
            atom2 = bond.GetEndAtomIdx()

            # 删除化学键
            rwmol.RemoveBond(atom1, atom2)
        
        # 生成新的SMILES
        try:
            new_smiles = Chem.MolToSmiles(rwmol)
            augmented_smiles.add(new_smiles)
        except Exception as e:
            print(f"Error generating SMILES: {e}")
            continue
    
    return list(augmented_smiles)


def get_equivalent_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, doRandom=True)
    return smiles

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
        esm2_output_conv = esm2_output_conv.squeeze(1)#torch.Size([32, 640])

        gcn_output = self.gcn_model(graph)#([32, 640])

        combined_features = torch.cat((esm2_output_conv, gcn_output), dim=1)
        #torch.Size([32, 1280])
        combined_features = combined_features.unsqueeze(1)#torch.Size([32, 1, 1280])

        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
        attn_output = attn_output.squeeze(1)#torch.Size([32, 1280])

        x = combined_features.squeeze(1) + attn_output #torch.Size([32, 1280])

        x = F.relu(self.fc1(x))#torch.Size([32, 256])
        x = F.relu(self.fc2(x))#torch.Size([32, 128])

        logits = self.classifier(self.dropout(x))
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, 2), labels.view(-1))
            outputs = (loss,) + (logits,)
            return outputs
        else:
            raise ('no label')

class CombinedDataset(Dataset):
    def __init__(self, csv_file, esm2_model_name, max_length=64, augment=False):
        self.protein_data = pd.read_csv(csv_file)
        
        self.esm2_tokenizer = AutoTokenizer.from_pretrained(esm2_model_name)
        
        self.max_length = max_length
        self.augment = augment

        self.graph_data = datasets.load_dataset("csv", cache_dir='./__pycache__',
                                                data_files=csv_file)
        self.graph_data = self.graph_data.map(lambda x: {"SMILES": x["SMILES"], "label": x["label"]})

    def __len__(self):
        return len(self.protein_data)

    def __getitem__(self, idx):
        sequence = self.protein_data.iloc[idx]['sequence']
        label = self.protein_data.iloc[idx]['label']
        partition=self.protein_data.iloc[idx]['partition']
        original_sequence = sequence
        if self.augment:
            sequence = augment_sequence(sequence)
            if random.random() < 0.5:
                sequence = reverse_sequence(sequence)

        esm2_encoded = self.esm2_tokenizer(sequence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        smiles = self.graph_data['train'][int(idx)]["SMILES"]

        if self.augment:
            smiles = random.choice(augment_smiles(smiles) + [smiles])
            smiles = get_equivalent_smiles(smiles)

        graph = smiles_to_graph(smiles)
        
        item = {
            'original_sequence': original_sequence,  # 返回原始序列
            'sequence': sequence,
            'esm2_input_ids': esm2_encoded['input_ids'].squeeze(0),
            'esm2_attention_mask': esm2_encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'graph': graph,
            'partition':partition
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
    print(f"Training fold {fold + 1}")
    torch.cuda.empty_cache()
    
    mymodel = CombinedModel(esm2_model_name=esm2_model_path)
    mymodel = mymodel.cuda()

    train_idx = df[df['partition'] != fold].index.tolist()
    test_idx = df[df['partition'] == fold].index.tolist()
    train_dataset = CombinedDataset(csv_file=data_file_path,
                                    esm2_model_name=esm2_model_path, 
                                    augment=False)
    train_noag = CombinedDataset(csv_file=data_file_path, 
                                 esm2_model_name=esm2_model_path, 
                                 augment=False)
    test_dataset = CombinedDataset(csv_file=data_file_path, 
                                   esm2_model_name=esm2_model_path, 
                                   augment=False)

    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    train_noag_sub = torch.utils.data.Subset(train_noag, train_idx)
    test_subset = torch.utils.data.Subset(test_dataset, test_idx)

    # 获取每个子集的原始序列
    train_sequences = [train_subset[i]['original_sequence'] for i in range(len(train_subset))]
    train_noag_sequences = [train_noag_sub[i]['original_sequence'] for i in range(len(train_noag_sub))]
    test_sequences = [test_subset[i]['original_sequence'] for i in range(len(test_subset))]

    # 获取原始的划分数据
    df_train_sequences = df[df['partition'] != fold]['sequence'].tolist()
    df_test_sequences = df[df['partition'] == fold]['sequence'].tolist()

    # 检查训练数据集的原始序列
    train_consistent = set(train_sequences) == set(df_train_sequences)
    train_noag_consistent = set(train_noag_sequences) == set(df_train_sequences)
    train_noag_train_consistent = set(train_noag_sequences) == set(train_sequences)
    test_consistent = set(test_sequences) == set(df_test_sequences)

    # 检查各数据集之间是否有重复
    test_train_overlap = set(test_sequences).intersection(set(train_sequences))
    test_train_noag_overlap = set(test_sequences).intersection(set(train_noag_sequences))

    print(f"Train subset consistent: {train_consistent}")
    print(f"Train no-augment subset consistent: {train_noag_consistent}")
    print(f"Train no-augment & Train subset consistent: {train_noag_train_consistent}")
    print(f"Test subset consistent: {test_consistent}")

    if test_train_overlap or test_train_noag_overlap:
        print("Test subset overlaps with train subset or train no-augment subset!")
    else:
        print("No overlap between test subset and train subsets.")

    if not train_consistent or not train_noag_consistent or not test_consistent or not train_noag_train_consistent:
        print("Data subsets do not match the original partition!")
    else:
        print("Data subsets match the original partition correctly.")

    assert train_consistent, "Train subset does not match the original partition!"
    assert train_noag_consistent, "Train no-augment subset does not match the original partition!"
    assert test_consistent, "Test subset does not match the original partition!"
    assert train_noag_train_consistent, "Train no-augment & Train subset do not match the original partition!"
    assert not test_train_overlap, "Test subset overlaps with train subset!"
    assert not test_train_noag_overlap, "Test subset overlaps with train no-augment subset!"

    outPutDir = out_file_path+f'/output{k}_fold{fold + 1}'

    training_args = TrainingArguments(
        output_dir=str(outPutDir),
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_ratio=0.2,
        weight_decay=0.01,
        learning_rate=(2e-5),
        logging_dir=str(outPutDir) + os.sep + 'logs',
        evaluation_strategy='steps',
        eval_steps=0.2,
        logging_steps=50,
        save_strategy='steps',
        save_steps=0.2,
        save_total_limit=3,
        seed=702,
        lr_scheduler_type='cosine_with_restarts',#'constant',#"cosine_with_restarts",
        optim='adamw_hf',#'adamw_hf',
        dataloader_drop_last=False,
        remove_unused_columns=False,
        load_best_model_at_end=False,
        metric_for_best_model='ACC',
    )

    trainer = Trainer(
        model=mymodel,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=test_subset,
        compute_metrics=Getmetrics.getMetrics,
        data_collator=custom_collate_fn, 
    )

    trainer.train()
    
    train_subset_data = [{'sequence': item['original_sequence'], 'label': item['labels'].item(), 'partition': item['partition']} for item in train_subset]
    train_noag_subset_data = [{'sequence': item['original_sequence'], 'label': item['labels'].item(), 'partition': item['partition']} for item in train_noag_sub]
    test_subset_data = [{'sequence': item['original_sequence'], 'label': item['labels'].item(), 'partition': item['partition']} for item in test_subset]
    
    # 转换为 DataFrame
    train_subset_df = pd.DataFrame(train_subset_data)
    train_noag_subset_df = pd.DataFrame(train_noag_subset_data)
    test_subset_df = pd.DataFrame(test_subset_data)

    train_subset_path = outPutDir+os.sep+'train_subset.csv'
    train_noag_subset_path = outPutDir+os.sep+'train_noag_subset.csv'
    test_subset_path = outPutDir+os.sep+'test_subset.csv'
    
    # 保存为 CSV
    train_subset_df.to_csv(train_subset_path, index=False)
    train_noag_subset_df.to_csv(train_noag_subset_path, index=False)
    test_subset_df.to_csv(test_subset_path, index=False)

    print("CSV 文件保存成功。")
    
    predictions = trainer.predict(test_subset)
    metrics = Getmetrics.getMetrics((predictions.predictions, predictions.label_ids))
    all_fold_metrics.append(metrics)

    print('*'*20)
    print(f'FOLD{fold+1}:')
    print(metrics)
    print('*'*20)
    
    logits_numpy = np.array(predictions.predictions)
    
    res_dict = {
        'sequence': [mydataset[i]['sequence'] for i in test_idx],
        'partition':[mydataset[i]['partition'] for i in test_idx],
        'score': [Getmetrics.getScore(logits_numpy[i]) for i in range(len(logits_numpy))],
        'predict_label': [Getmetrics.getPredictLabel(logits_numpy[i]) for i in range(len(logits_numpy))],
        'y_label': predictions.label_ids,
    }
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv(outPutDir + os.sep + 'predict_res.csv', index=None)

    res_met_df = pd.DataFrame(metrics, index=[1]).T
    res_met_df.to_csv(outPutDir + os.sep + 'metrics_res.csv')
    # del mymodel
    # del trainer
mean_metrics = {metric: np.mean([fold_metrics[metric] for fold_metrics in all_fold_metrics]) for metric in all_fold_metrics[0]}

print("Mean metrics:", mean_metrics)

all_fold_metrics_df = pd.DataFrame(all_fold_metrics)
mean_metrics_df = pd.DataFrame([mean_metrics], index=['mean'])

final_metrics_df = pd.concat([all_fold_metrics_df, mean_metrics_df])
final_metrics_df.to_csv(res_path)
