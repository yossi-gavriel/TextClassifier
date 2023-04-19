from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        """
        TextDataset constructor.

        Args:
            texts (list[str]): A list of text inputs to the model.
            labels (list[int]): A list of corresponding labels for each input text.
            tokenizer (transformers.Tokenizer): A tokenizer instance from the transformers library.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        """
        Gets the item at a given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input_ids, attention_mask, and label for the item.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=256)
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'label': torch.tensor(label)
        }

    def __len__(self):
        """
        Gets the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.labels)