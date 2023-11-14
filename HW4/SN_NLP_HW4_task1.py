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
class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=256, linear_dim=128, dropout=0.33, num_layers=1):
        super(BiLSTMNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first = True,
                            bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim, linear_dim)
        self.elu = nn.ELU(alpha = 0.75)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(linear_dim, tagset_size)

    def forward(self, sentence, lengths):
        embedded = self.embedding(sentence)

        # Pack the embeddings
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_lstm_out, _ = self.lstm(packed_embedded)

        # Unpack the sequence
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)
        linear_out = self.elu(self.linear(lstm_out))
        tag_space = self.classifier(linear_out)

        return tag_space.permute(0,2,1)

def preprocess_data(data):
    input_ids = [torch.tensor(seq) for seq in data['input_ids']]
    labels = [torch.tensor(label) for label in data['labels']]
    return list(zip(input_ids, labels))

def dynamic_padding(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = torch.tensor([len(inp) for inp in inputs])

    # Dynamic padding in the batch
    inputs = pad_sequence(inputs, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)

    return inputs, labels, lengths

def get_predictions(model, loader, device):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for inputs, _, lengths in loader: # We don't need targets now
            inputs = inputs.to(device)

            outputs = model(inputs, lengths)
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)

            # Truncate predictions to their original lengths
            truncated_predictions = [pred[:len_].tolist() for pred, len_ in zip(predictions, lengths)]

            all_predictions.extend(truncated_predictions)

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
BATCH_SIZE = 32
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
model = BiLSTMNER(vocab_size,tagset_size)
model.load_state_dict(torch.load('model_hw4_task1.pth'))
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