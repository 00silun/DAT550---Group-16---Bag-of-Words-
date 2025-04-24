from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader

def prepare_data(df, model_type='FFNN', vocab_size=10000, max_seq_len=200, batch_size=64, use_tfidf=False):
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])

    if model_type.upper() == 'FFNN':
        vectorizer_class = TfidfVectorizer if use_tfidf else CountVectorizer
        vectorizer = vectorizer_class(max_features=vocab_size)
        X = vectorizer.fit_transform(df['processed_abstract'])
        y = df['label_encoded']

        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

        def to_tensor_dataset(X, y):
            return TensorDataset(
                torch.tensor(X.toarray(), dtype=torch.float32),
                torch.tensor(y.values, dtype=torch.long)
            )

        train_loader = DataLoader(to_tensor_dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(to_tensor_dataset(X_val, y_val), batch_size=batch_size)
        X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
        input_dim = X.shape[1]

        return train_loader, val_loader, X_test_tensor, y_test_tensor, input_dim, len(label_encoder.classes_), None

    elif model_type.upper() == 'RNN':
        tokenized = [t.lower().split() for t in df['processed_abstract']]

        all_words = [word for sent in tokenized for word in sent]
        most_common = Counter(all_words).most_common(vocab_size - 2)
        vocab = {w: i + 2 for i, (w, _) in enumerate(most_common)}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1

        def encode(tokens):
            ids = [vocab.get(word, 1) for word in tokens]
            return ids[:max_seq_len] + [0] * max(0, max_seq_len - len(ids))

        X = [encode(text.lower().split()) for text in df['processed_abstract']]

        y = df['label_encoded'].tolist()

        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

        def make_loader(X, y):
            return DataLoader(TensorDataset(
                torch.tensor(X, dtype=torch.long),
                torch.tensor(y, dtype=torch.long)
            ), batch_size=batch_size, shuffle=True)

        train_loader = make_loader(X_train, y_train)
        val_loader = make_loader(X_val, y_val)
        X_test_tensor = torch.tensor(X_test, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        return train_loader, val_loader, X_test_tensor, y_test_tensor, vocab_size, len(label_encoder.classes_), vocab
