import lightning as L
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import T5Tokenizer


class TextDataset(Dataset):
    def __init__(self,  max_length, csv_filepath, token_filepath):
        super(TextDataset, self).__init__()
        self.max_length = max_length

        self.filename = pd.read_csv(csv_filepath)

        with open(token_filepath, "r") as outfile:
            local_tokens = outfile.read()

        local_tokens = local_tokens.split("\n")
        self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        self.tokenizer.add_tokens(local_tokens)

        self.tokenizer.save_pretrained("saved_weights/tokenizer")

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):

        input_tokenize = self.tokenizer(
            str(self.filename.loc[idx, 'source']),
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        output_tokenize = self.tokenizer(
            str(self.filename.loc[idx, 'target']),
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = input_tokenize['input_ids'].flatten()
        attention_mask = input_tokenize['attention_mask'].flatten()
        labels = output_tokenize['input_ids'].flatten()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class TextDataLoader(L.LightningDataModule):
    def __init__(self, max_length, batch_size, csv_filepath, token_filepath):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.csv_filepath = csv_filepath
        self.token_filepath = token_filepath

    def setup(self, stage=None):
        self.dataset = TextDataset(self.max_length, self.csv_filepath, self.token_filepath)

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          num_workers=11,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          num_workers=11,
                          shuffle=False)
"""
debugging stuff
"""
if __name__ == "__main__":
    # dataset = TextDataset(max_length=32,
    #                       csv_filepath="../data/processed/processed_data.csv",
    #                       token_filepath="../data/processed/tokens.txt")
    # print(len(dataset))
    # print(next(iter(dataset)))
    dataloader = TextDataLoader(max_length=32,
                                batch_size=8,
                                csv_filepath="../data/processed/processed_data.csv",
                                token_filepath="../data/processed/tokens.txt")
    dataloader.setup()
    print(next(iter(dataloader.train_dataloader())))

