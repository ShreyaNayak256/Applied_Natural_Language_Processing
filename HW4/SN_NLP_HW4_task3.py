import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import datasets
import itertools
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
# assuming the conlleval python file is in the same directory as the python file
from conlleval import evaluate
import itertools
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerNER(nn.Module):
    def __init__(self, vocab_size, tagset_size, emb_size=128, nhead=8, ff_size=512, num_layers=6, dropout=0.1, max_seq_length=128):
        super(TransformerNER, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=ff_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.classifier = nn.Linear(emb_size, tagset_size)
        self.emb_size = emb_size
    def forward(self, src, src_mask):
        src = self.token_embedding(src) * math.sqrt(self.emb_size)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        logits = self.classifier(output)
        return logits

def preprocess_data(data):
    input_ids = [torch.tensor(seq) for seq in data['input_ids']]
    labels = [torch.tensor(label) for label in data['labels']]
    return list(zip(input_ids, labels))

def dynamic_padding(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # Dynamic padding in the batch
    inputs_padded = pad_sequence(inputs, batch_first=False)  # Transformer expects (seq_len, batch, feature)
    labels_padded = pad_sequence(labels, batch_first=False, padding_value=-1)  # padding as -1
    # Create the source mask for the transformer
    # `True` values are where the attention should NOT focus (i.e., padding)
    src_mask = (inputs_padded == 0).transpose(0, 1)

    return inputs_padded, labels_padded, src_mask

def get_predictions(model, loader, device):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for inputs, _, src_mask in loader: # We don't need targets, but we do need the mask
            inputs = inputs.to(device)
            src_mask = src_mask.to(device)

            # Forward pass, get logits for each token in the sequence
            outputs = model(inputs, src_mask)

            # Get predictions
            predictions = torch.argmax(outputs, dim=2)  # dim=2 because outputs are (seq_length, batch, num_tags)

            # Transpose predictions to match inputs shape
            predictions = predictions.transpose(0, 1)  # Now predictions are (batch, seq_length)

            # Remove padding (convert masks to indices and select non-padded elements)
            for batch_idx, batch in enumerate(predictions):
                # Get the indices where src_mask is False (meaning valid tokens, not padding)
                valid_indices = ~src_mask[batch_idx]
                valid_predictions = batch[valid_indices]
                all_predictions.append(valid_predictions.tolist())

    return all_predictions

print("Loading Dataset: ")
dataset = datasets.load_dataset("conll2003")

word_frequency = Counter(itertools.chain(*dataset['train']['tokens']))  # type: ignore
# Remove words below threshold 2
word2idx = {
    word: frequency
    for word, frequency in word_frequency.items()
    if frequency >= 2
}

word2idx = {
    word: index
    for index, word in enumerate(word_frequency.keys(), start=2)
}
word2idx['[PAD]'] = 0
word2idx['[UNK]'] = 1
print("Processing Dataset....")
dataset = (
    dataset
    .map(lambda x: {
            'input_ids': [
                word2idx.get(word, word2idx['[UNK]'])
                for word in x['tokens']
            ]
        }
    )
)
columns_to_remove = ['pos_tags', 'chunk_tags']
for split in dataset.keys():
    dataset[split] = dataset[split].remove_columns(columns_to_remove)

# Rename ner_tags to labels
for split in dataset.keys():
    dataset[split] = dataset[split].rename_column('ner_tags', 'labels')

label2id = dataset["train"].features["labels"].feature
id2label = {id: label for label, id in enumerate(label2id.names)}
# label2id.names
id2label['PAD'] = -1

print("dataset processed. Initiating dataloaders")
# Hyperparameters
BATCH_SIZE = 4
# Preprocess the train, val, and test data
train_data = preprocess_data(dataset['train'])
val_data = preprocess_data(dataset['validation'])
test_data = preprocess_data(dataset['test'])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dynamic_padding)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=dynamic_padding)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=dynamic_padding)

vocab_size = max([max(seq) for seq in dataset['train']['input_ids']]) + 1
tagset_size = max([max(seq) for seq in dataset['train']['labels']]) + 1
# Device definition
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_labels = [label for sublist in dataset['train']['labels'] for label in sublist]
label_counts = {label: all_labels.count(label) for label in id2label.values() if label != -1}
total_labels = len(all_labels) # We subtract the count of 'PAD' labels
weights = [total_labels / label_counts[id2label[key]] for key in id2label.keys() if key != 'PAD']
# Normalization idea 1: dividing by max weight
# max_weight = max(weights)
# weights = [weight / max_weight for weight in weights]
# Normalization idea 2: dividing by sum of weights
weights = [weight / sum(weights) for weight in weights]

weights_tensor = torch.tensor(weights).to(device)
 # torch loading code
print("Loading model...")
model = TransformerNER(vocab_size,tagset_size)
model.load_state_dict(torch.load('model_hw4_task3.pth'))
print("Beginning predictions...")
val_predictions = get_predictions(model, val_loader, device)
test_predictions = get_predictions(model, test_loader, device)

# print("Validation Set: ")
# idx2tag  = {id:tag for (tag,id) in id2label.items()}
# labels = [
# list(map(idx2tag.get, labels))
# for labels in dataset['validation']['labels']
# ]
# preds = [
# list(map(idx2tag.get, labels))
# for labels in val_predictions
# ]
# precision, recall, f1 = evaluate(itertools.chain(*labels),itertools.chain(*preds))
print("Test Set: ")
idx2tag  = {id:tag for (tag,id) in id2label.items()}
labels = [
list(map(idx2tag.get, labels))
for labels in dataset['test']['labels']
]
preds = [
list(map(idx2tag.get, labels))
for labels in test_predictions
]
precision, recall, f1 = evaluate(itertools.chain(*labels),itertools.chain(*preds))