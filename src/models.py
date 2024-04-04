import torch
import lightning as L
from transformers import T5ForConditionalGeneration


class T5Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        loss, _ = self.forward(input_ids, attention_mask, labels)

        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        loss, _ = self.forward(input_ids, attention_mask, labels)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)










