import torch

import dataset
import models.bart_classifier
import models.prototex

dataset_file = 'dataset/train/chat.txt'
label_file = 'dataset/train/labels.txt'
classes_file = 'dataset/class_names.txt'

print("Loading data...")
TINY = False
train_dataset = dataset.TwitchEmoteDataset(dataset_file, label_file, classes_file,
                                           tokenizer=models.bart_classifier.BartClassifier.get_tokenizer(), tiny=TINY)
val_dataset = dataset.TwitchEmoteDataset('dataset/val/chat.txt', 'dataset/val/labels.txt', classes_file,
                                         tokenizer=models.bart_classifier.BartClassifier.get_tokenizer(), tiny=TINY)
test_dataset = dataset.TwitchEmoteDataset('dataset/test/chat.txt', 'dataset/test/labels.txt', classes_file,
                                          tokenizer=models.bart_classifier.BartClassifier.get_tokenizer(), tiny=TINY)
print("Done")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024,
                                           collate_fn=dataset.TwitchEmoteDataset.collate_fn,
                                           sampler=dataset.BalancedSampler(train_dataset), drop_last=True)
train_eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024,
                                                collate_fn=dataset.TwitchEmoteDataset.collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, collate_fn=dataset.TwitchEmoteDataset.collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024,
                                          collate_fn=dataset.TwitchEmoteDataset.collate_fn)
params = models.prototex.Prototex.get_default_params()

models.prototex.Prototex.train_model(train_loader, val_loader, test_loader, train_eval_loader,
                                                  params=params, num_classes=len(train_dataset.classes),
                                                 class_names=train_dataset.classes)

