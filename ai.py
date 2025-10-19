import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model/dog_breed_efficientnet (805).pt"
IMG_SIZE = 256

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model():
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 120)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


def predict_breed(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_t)
        probs = F.softmax(logits, dim=1)
        top3 = torch.topk(probs, 3)
    probs, indices = top3.values.squeeze(), top3.indices.squeeze()
    return [(int(i), float(p)) for i, p in zip(indices, probs)]
