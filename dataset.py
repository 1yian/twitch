import numpy as np
import torch


class TwitchEmoteDataset(torch.utils.data.Dataset):

    def __init__(self, text_file, label_file, classes_file, tokenizer, num_classes=64, tiny=False):
        super().__init__()
        input_texts = TwitchEmoteDataset._read_file_into_list(text_file)
        labels = TwitchEmoteDataset._read_file_into_list(label_file)
        self.classes = TwitchEmoteDataset._read_file_into_list(classes_file)[:num_classes]
        self.num_classes = num_classes
        labels = [int(i) for i in labels]
        input_texts_final = []
        labels_final = []
        if tiny:
            labels = labels[:3000]
        for i in range(len(labels)):
            if labels[i] < num_classes:
                input_texts_final.append(input_texts[i])
                labels_final.append(labels[i])

        self.labels = labels_final
        input_texts = input_texts_final
        self.input_ids, self.attn_masks = [], []
        encoding = tokenizer([string.split() for string in input_texts], padding='max_length',
                             truncation=True, max_length=32, return_attention_mask=True, is_split_into_words=True)
        inp_id, attn_mask = encoding['input_ids'], encoding['attention_mask']
        self.input_ids += inp_id
        self.attn_masks += attn_mask

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        return (torch.LongTensor([list(i[0]) for i in batch]), torch.Tensor([list(i[1]) for i in batch]),
                torch.LongTensor([i[2] for i in batch]))

    @staticmethod
    def _read_file_into_list(path):
        lines = []
        with open(path, encoding='utf-8') as file:
            for line in file:
                lines.append(line.strip())
        return lines


class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_samples=None):
        class_weights = np.zeros(dataset.num_classes)
        for idx in range(len(dataset)):
            _, _, class_idx = dataset[idx]
            class_idx = int(class_idx)
            class_weights[class_idx] += 1
        self.num_samples = num_samples if num_samples is not None else len(dataset)

        self.idx_weights = []
        for instance in dataset:
            _, _, class_idx = instance
            self.idx_weights.append(1. / class_weights[class_idx])
        self.class_weights = class_weights

    def __iter__(self):
        return iter(torch.multinomial(torch.Tensor(self.idx_weights), self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
