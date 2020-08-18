from src.data import parse_faces
from src.model import CoupleFaceNet
import src.config as config
from glob import glob
import os
import torch
from PIL import Image
import random
from torchvision import transforms


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]
)


def predict(show_imge=False):
    faces = parse_faces('inference', 'inference', predict=True)
    random_cl1_anchors = torch.stack([transform(
        Image.open(img)
    ) for img in random.sample(
        glob(os.path.join('faces', 'crab', '*')), 32
    )]).to(config.DEVICE)
    random_cl2_anchors = torch.stack([transform(
        Image.open(img)
    ) for img in random.sample(
        glob(os.path.join('faces', 'duck', '*')), 32
    )]).to(config.DEVICE)
    model = CoupleFaceNet().to(config.DEVICE)
    model.load_state_dict(torch.load('model_weights/face_reid.pt'))
    model.eval()

    cl1_embed = model(random_cl1_anchors).mean(dim=0)
    cl2_embed = model(random_cl2_anchors).mean(dim=0)

    for face in faces:
        face_embed = model(transform(face).unsqueeze(0).to(config.DEVICE))
        class1_dist = torch.norm(cl1_embed - face_embed)
        class2_dist = torch.norm(cl2_embed - face_embed)
        _, index = torch.tensor([class1_dist.item(),
                                 class2_dist.item()]).min(dim=0)
        if show_imge:
            face.show()
        print(f'This face belongs to class {index.item()}')


if __name__ == '__main__':
    predict()
