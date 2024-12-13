import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torch.distributions import Categorical
from transformers import AutoTokenizer

from collections import OrderedDict

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model_implement.transformer import EncoderDecoder  # Your custom model definition


def load_model(model_path, device, image_size, val_images, tokenizer):
    """Load the model and set to evaluation mode."""
    model = EncoderDecoder(image_size=image_size, channels_in=val_images.shape[1],
                                     num_emb=tokenizer.vocab_size, patch_size=8,
                                     num_layers=(6, 6), hidden_size=192,
                                     num_heads=8)

    state_dict = torch.load(model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove 'module.' prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    return model

def preprocess_image(image_path, image_size):
    """Preprocess the input image."""
    transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def generate_caption(model, tokenizer, image_tensor, device):
    """Generate captions from the model."""

    # Add the Start-Of-Sentence token to the prompt to signal the network to start generating the caption
    sos_token = 101 * torch.ones(1, 1).long()

    # Set the temperature for sampling during generation
    temp = 0.5

    log_tokens = [sos_token]

    with torch.no_grad():
        # Encode the input image
        with torch.cuda.amp.autocast():
            # Forward pass
            image_embedding = model.encoder(image_tensor.to(device))

        # Generate the answer tokens
        for i in range(50):
            input_tokens = torch.cat(log_tokens, 1)

            # Decode the input tokens into the next predicted tokens
            data_pred = model.decoder(input_tokens.to(device), image_embedding)

            # Sample from the distribution of predicted probabilities
            dist = Categorical(logits=data_pred[:, -1] / temp)
            next_tokens = dist.sample().reshape(1, 1)

            # Append the next predicted token to the sequence
            log_tokens.append(next_tokens.cpu())

            # Break the loop if the End-Of-Caption token is predicted
            if next_tokens.item() == 102:
                break

    # Convert the list of token indices to a tensor
    pred_text = torch.cat(log_tokens, 1)

    # Convert the token indices to their corresponding strings using the vocabulary
    pred_text_strings = tokenizer.decode(pred_text[0], skip_special_tokens=True)

    # Join the token strings to form the predicted text
    pred_text = "".join(pred_text_strings)
    
    return pred_text  

def main(image_path, image_size = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    model_checkpoint = r"../model/transf_model.pth"

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Preprocess input image
    image_tensor = preprocess_image(image_path, image_size).to(device)

    model = load_model(model_checkpoint, device, image_size, image_tensor, tokenizer)

    # Generate caption
    caption = generate_caption(model, tokenizer, image_tensor, device)
    print(f"Generated Caption: {caption}")


if __name__ == "__main__":
    image_path = r"../000000000001.jpg"

    main(image_path)
