from src.dataloaders import TextDataLoader
from src.models import T5Model
from src.utils import CustomTrainer
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
import os
from pathlib import Path
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = "1"




if __name__ == "__main__":
    csv_filepath = "data/processed/processed_data.csv"
    token_filepath = "data/processed/tokens.txt"
    dataloader = TextDataLoader(max_length=32, batch_size=256, csv_filepath=csv_filepath, token_filepath=token_filepath)
    dataloader.setup()
    root_dir = Path(os.getcwd())
    model = T5Model(str(root_dir / "saved_weights/checkpoints")).to(DEVICE)
    model.model.resize_token_embeddings(len(dataloader.dataset.tokenizer))  #### increase the token enbeddings

    checkpoint = ModelCheckpoint(
        dirpath="saved_weights/checkpoints",
        # filename="saved_model",
        monitor="val_loss",
    )

    trainer = CustomTrainer(
        callbacks=[checkpoint],
        max_epochs=50,
        accelerator="gpu"
    )

    trainer.fit(model, dataloader)
