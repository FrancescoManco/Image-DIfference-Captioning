import torch
import clip
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Carichiamo il modello CLIP congelato
class FrozenCLIPEncoder(nn.Module):
    def __init__(self, model_name='ViT-B/32'):
        super(FrozenCLIPEncoder, self).__init__()
        self.clip_model, _ = clip.load(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.eval()  # Congeliamo l'encoder CLIP

    def forward(self, image):
        with torch.no_grad():  # Congeliamo i pesi di CLIP
            image_features = self.clip_model.encode_image(image)
        return image_features
    
class DifferenceProcessor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DifferenceProcessor, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, output_dim)  # Concatenazione di due embeddings
        self.layernorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, feature1, feature2):
        combined_features = torch.cat((feature1, feature2), dim=1)  # Concatenazione
        difference_features = self.fc1(combined_features)
        difference_features = self.layernorm(difference_features)
        difference_features = self.dropout(difference_features)
        return difference_features

class FeatureProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureProjection, self).__init__()
        self.linear_proj = nn.Linear(input_dim, output_dim)  # Proiezione Lineare

    def forward(self, features):
        return self.linear_proj(features)

class GPT2Decoder(nn.Module):
    def __init__(self, gpt2_model_name="gpt2"):
        super(GPT2Decoder, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

    def forward(self, difference_features, target_caption):
        # Prepariamo l'input per il GPT-2
        inputs = self.tokenizer(target_caption, return_tensors="pt").input_ids.to(difference_features.device)

        # Generiamo la caption con cross-attention usando le difference features come contesto
        outputs = self.gpt2(inputs, encoder_hidden_states=difference_features.unsqueeze(1))
        return outputs.logits

class ImageDifferenceCaptioningModel(nn.Module):
    def __init__(self, clip_model_name='ViT-B/32', gpt2_model_name="gpt2"):
        super(ImageDifferenceCaptioningModel, self).__init__()
        self.encoder = FrozenCLIPEncoder(model_name=clip_model_name)
        self.difference_processor = DifferenceProcessor(input_dim=512, output_dim=768)
        self.feature_projection = FeatureProjection(input_dim=768, output_dim=768)
        self.decoder = GPT2Decoder(gpt2_model_name=gpt2_model_name)

    def forward(self, image1, image2, target_caption):
        # Otteniamo le features delle immagini da CLIP
        feature1 = self.encoder(image1)
        feature2 = self.encoder(image2)
        
        # Elaborazione della differenza
        difference_features = self.difference_processor(feature1, feature2)
        
        # Proiezione delle features
        projected_features = self.feature_projection(difference_features)
        
        # Decodifica della caption delle differenze
        output_logits = self.decoder(projected_features, target_caption)
        
        return output_logits


criterion = nn.CrossEntropyLoss()


def train_step(model, image1, image2, caption_difference):
    model.train()
    
    # Calcola l'output del modello
    logits = model(image1, image2, caption_difference)
    
    # Calcola la perdita
    targets = model.decoder.tokenizer(caption_difference, return_tensors="pt").input_ids.to(logits.device)
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Backpropagation e ottimizzazione
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()
