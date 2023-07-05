import torch
import clip
import os
from torchvision.datasets import CIFAR10
import numpy as np

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)

# from https://github.com/openai/CLIP/blob/main/data/prompts.md
cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck',]
cifar10_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

class_map = {'CIFAR10': cifar10_classes}
template_map = {'CIFAR10': cifar10_templates}

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

@torch.no_grad()
def extract_text_features(dataset_name):
    # code borrowed from: https://github.com/openai/CLIP/blob/fcab8b6eb92af684e7ff0a904464be7b99b49b88/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    class_names = class_map[dataset_name]
    templates = template_map[dataset_name]
    model.to(device)
    model.eval()

    zeroshot_weights = []
    for classname in class_names:
        texts = [template.format(classname) for template in templates]
        texts = clip.tokenize(texts).to(device)
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

cifar10 = CIFAR10(root=os.path.expanduser("data"), download=True, train=False)

for dataset in [cifar10]:
    print(dataset)
    # extract image feature, code borrowed from: https://github.com/openai/CLIP#zero-shot-prediction
    image_features = []
    image_labels = []
    for image, class_id in dataset:
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
        image_feature /= image_feature.norm()
        image_features.append(image_feature)
        image_labels.append(class_id)
    image_features = torch.stack(image_features, dim=1).to(device)
    image_features = image_features.squeeze()
    
    # extract text feature
    dataset_name = 'CIFAR10'
    text_features = extract_text_features(dataset_name)
    
    # compute top-1 accuracy
    logits = (100. * image_features @ text_features).softmax(dim=-1)
    image_labels = torch.tensor(image_labels).unsqueeze(dim=1).to(device)
    top1_acc = accuracy(logits, image_labels, (1,))
    print(f'top-1 accuracy for {dataset_name} dataset: {top1_acc[0]:.3f}')
