from facenet_pytorch import MTCNN
import torch
from glob import glob
import os
import PIL
from PIL import ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FaceData(Dataset):
    def __init__(self, seed=0, data_path='faces', transforms=True):
        random.seed(seed)
        self.transforms = transforms
        self.class2idx = {os.path.split(f.path)[-1]: i for i, f in enumerate(
            os.scandir(data_path)
        ) if f.is_dir()}
        self.idx2class = {i: class_ for class_, i in self.class2idx.items()}
        self.couple1 = self.idx2class[0]
        self.couple2 = self.idx2class[1]
        self.couple1_files = glob(os.path.join(data_path, self.couple1, '*'))
        self.couple2_files = glob(os.path.join(data_path, self.couple2, '*'))
        self.couple1_images = {i: file_path for i, file_path in enumerate(
            random.sample(self.couple1_files, len(self.couple1_files))
        )}
        self.couple2_images = {i: file_path for i, file_path in enumerate(
            random.sample(self.couple2_files, len(self.couple2_files))
        )}

    def _get_transforms(self):
        if self.transforms:
            transform = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ]
            )

        return transform

    def __len__(self):
        return min(len(self.couple1_images), len(self.couple2_images))

    def __getitem__(self, index):
        transform = self._get_transforms()
        item = {}
        item[self.idx2class[0]] = transform(
            PIL.Image.open(self.couple1_images[index])
        )
        item[self.idx2class[1]] = transform(
            PIL.Image.open(self.couple2_images[index])
        )
        item['target'] = [0, 1]
        return item


def get_loaders(seed, shuffle=True, batch_size=64):
    """Image dataset loaders"""

    dataset = FaceData(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return loader


def parse_faces(data_path, outpath):
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(
        keep_all=True,
        image_size=224, margin=5, min_face_size=10,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        device=device
    )

    imgs = glob(os.path.join(data_path, '*'))[149:]

    for img_path in tqdm(imgs, total=len(imgs)):
        print(img_path)
        try:
            img = PIL.Image.open(img_path).convert('RGB')
            boxes, probs = mtcnn.detect(img)
        except PIL.UnidentifiedImageError:
            continue
        if boxes is None:
            continue
        for box, prob, i in zip(boxes, probs, range(len(probs))):
            if prob < 0.9:
                continue
            face = img.crop(box)
            img_name = img_path.split('/')[-1]
            face.save(os.path.join(
                outpath,
                img_name.split('.')[0] + str(i) + '.' + img_name.split('.')[1])
            )


def resize_imgs(data_path):
    import numpy as np
    from tqdm import tqdm
    for class_ in glob(os.path.join(data_path, '*')):
        for img_path in tqdm(glob(os.path.join(class_, '*'))):
            img = PIL.Image.open(img_path)
            if np.array(img).shape == (224, 224):
                continue
            img = img.resize((224, 224))
            img.save(img_path)


if __name__ == '__main__':
    parse_faces('data', 'faces')
    resize_imgs('faces')
