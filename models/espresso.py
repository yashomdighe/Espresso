import torch
import lightning as L

class Espresso(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs["tokenizer"]
        self.text_encoder = kwargs["text_encoder"]
        self.pcd_encoder = kwargs["pcd_encoder"]

        # TO DO 
        # Add attention layer
        # Setup and add the rasterization layer

        


