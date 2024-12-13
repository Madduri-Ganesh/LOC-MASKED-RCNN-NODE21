# %%
import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
import SimpleITK as sitk
import utils
import transforms as T
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
import torch
from torchvision.ops import box_iou
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from sklearn.metrics import roc_auc_score

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
data = pd.read_csv('/data/courses/2024/class_ImageSummerFall2024_jliang12/node21/cxr_images/proccessed_data/metadata.csv')

# %%
print(data.head)

# %%
print(data.columns)

for i in os.listdir('/data/courses/2024/class_ImageSummerFall2024_jliang12/node21/cxr_images/proccessed_data/images/'):
    print(i)
    break

# %%
n = 0
copied_filenames = []
source_dir = '/data/courses/2024/class_ImageSummerFall2024_jliang12/node21/cxr_images/proccessed_data/images/'
final_destination = '/home/mmaddur1/train_node21_rcnn/cxr_test/'
for each_file in tqdm(os.listdir(source_dir)):
    source_file = os.path.join(source_dir, each_file)
    destination_file = os.path.join(final_destination, each_file)
    if n == 976:
        break
    else:
        shutil.copyfile(source_file, destination_file)
        copied_filenames.append(each_file)
        n+=1

filtered_data = data[data['img_name'].isin(copied_filenames)]
filtered_data.to_csv('/home/mmaddur1/train_node21_rcnn/test_metadata.csv', index=False)

train_data = data[~data['img_name'].isin(copied_filenames)]
train_data.to_csv('/home/mmaddur1/train_node21_rcnn/train_metadata.csv', index=False)

# %%
print(len(copied_filenames))


# %%

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomCrop(0.5))
    return T.Compose(transforms)

class CXRNoduleDataset(object):
    def __init__(self, root, csv_file, transforms,train):
        self.root = root
        self.transforms = transforms
        self.data = pd.read_csv(csv_file)
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.imgs = [i for i in self.imgs if i[-9:] in self.data['img_name'].values]
        # Read only image files in following format
        self.imgs = [i  for i in self.imgs if os.path.splitext(i)[1].lower() in (".mhd", ".mha", ".dcm", ".png", ".jpg", ".jpeg")]
        if train:
            self.imgs = [i for i in self.imgs if i not in copied_filenames]
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", str(self.imgs[idx]))
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
     
        img = Image.fromarray((np.asarray(img)/np.max(img)))
        nodule_data = self.data[self.data['img_name']==str(self.imgs[idx])[-9:]]
        num_objs = len(nodule_data)
        boxes = []
        
        if nodule_data['label'].any()==1: # nodule data
            for i in range(num_objs):
                x_min = int(nodule_data.iloc[i]['x'])
                y_min = int(nodule_data.iloc[i]['y'])
                y_max = int(y_min+nodule_data.iloc[i]['height'])
                x_max = int(x_min+nodule_data.iloc[i]['width'])
                boxes.append([x_min, y_min, x_max, y_max])

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # for non-nodule images
        else:
            boxes = torch.empty([0,4])
            area = torch.tensor([0])
            labels = torch.zeros(0, dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

            
        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        image_name = str(self.imgs[idx])

        return img, target, image_name

    def __len__(self):
        return len(self.imgs)
    
    
input_dir = '/data/courses/2024/class_ImageSummerFall2024_jliang12/node21/cxr_images/proccessed_data/'  
test_input_dir = '/home/mmaddur1/train_node21_rcnn/cxr_test/'  
train_data = CXRNoduleDataset(input_dir, os.path.join(input_dir, 'metadata.csv'), get_transform(train=True), train=True)
test_data = CXRNoduleDataset(test_input_dir, os.path.join(test_input_dir, 'test_metadata.csv'), get_transform(train=False), train=False)



loaders = {
'train' : torch.utils.data.DataLoader(train_data,
batch_size=2,
shuffle=True,
num_workers=4,
collate_fn=utils.collate_fn),
'test' : torch.utils.data.DataLoader(test_data,
batch_size=2,
shuffle=True,
num_workers=4,
collate_fn=utils.collate_fn)
}

# %%


def get_model(num_classes, pretrain):
    # Load pre-trained Faster RCNN
    model = maskrcnn_resnet50_fpn(pretrained=pretrain)
    
    # Replace the classifier head with the number of classes in your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = None
    
    return model



def compute_froc(model, data_loader, iou_threshold=0.2, predefined_fppi=[0.125, 0.25, 0.5], device='cuda'):
    model.eval()
    all_image_scores = []
    all_image_labels = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                true_boxes = target['boxes'].cpu()

                # Assign image score (max probability or 0 if no predictions)
                image_score = pred_scores.max().item() if len(pred_scores) > 0 else 0
                all_image_scores.append(image_score)
                all_image_labels.append(1 if len(true_boxes) > 0 else 0)

                all_predictions.append((pred_boxes, pred_scores))
                all_targets.append(true_boxes)

    # Calculate AUC
    auc = roc_auc_score(all_image_labels, all_image_scores)

    # FROC analysis
    all_tp = []
    all_fp = []
    all_scores = []
    num_images = len(all_predictions)
    total_nodules = sum(len(target) for target in all_targets)

    for (pred_boxes, pred_scores), true_boxes in zip(all_predictions, all_targets):
        if len(pred_boxes) > 0 and len(true_boxes) > 0:
            ious = box_iou(pred_boxes, true_boxes)
            
            # For each ground truth, keep only the prediction with highest score and IoU > threshold
            max_iou, max_idx = ious.max(dim=0)
            valid_preds = max_iou >= iou_threshold

            for i, is_valid in enumerate(valid_preds):
                if is_valid:
                    all_tp.append(1)
                    all_fp.append(0)
                    all_scores.append(pred_scores[max_idx[i]].item())
                else:
                    all_tp.append(0)
                    all_fp.append(1)
                    all_scores.append(pred_scores[max_idx[i]].item())

            # Add remaining predictions as false positives
            remaining_preds = set(range(len(pred_boxes))) - set(max_idx[valid_preds].tolist())
            for i in remaining_preds:
                all_tp.append(0)
                all_fp.append(1)
                all_scores.append(pred_scores[i].item())
        else:
            # All predictions are false positives if there are no true boxes
            all_tp.extend([0] * len(pred_boxes))
            all_fp.extend([1] * len(pred_boxes))
            all_scores.extend(pred_scores.tolist())

    # Sort by confidence score
    sorted_indices = np.argsort(all_scores)[::-1]
    all_tp = np.array(all_tp)[sorted_indices]
    all_fp = np.array(all_fp)[sorted_indices]

    # Compute cumulative sums
    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(all_fp)

    sensitivities = cum_tp / total_nodules
    fppi = cum_fp / num_images

    # Calculate sensitivities at predefined FPPI rates
    froc_sensitivities = []
    for fp_rate in predefined_fppi:
        if fppi[-1] >= fp_rate:
            idx = np.searchsorted(fppi, fp_rate, side='right')
            froc_sensitivities.append(sensitivities[idx - 1])
        else:
            froc_sensitivities.append(0)

    # Plot FROC curve
    plt.figure()
    plt.plot(fppi, sensitivities)
    plt.xscale('log')
    plt.xlabel('False Positives Per Image (FPPI)')
    plt.ylabel('Sensitivity')
    plt.title('FROC Curve')
    plt.grid(True)
    plt.show()

    return auc, froc_sensitivities



# %%
with open('froc_values', 'w+') as f:
    for i in tqdm(range(10)):
        num_classes = 2  # NODE21 classes + 1 background
        model = get_model(num_classes, pretrain=False)
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)    

        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, targets, img_name in loaders['train']:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backprop
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(loaders['train'])}")

        auc, froc_sens  = compute_froc(model, loaders['test'])
        print(f"The FROC value for iteration {i} is {auc}")
        print(f"The FROC sensitivity for iteration {i} is {froc_sens}")
        f.write(f"The FROC auc value  and sensitivities for iteration {i} is {auc} and {froc_sens}\n")


    

# %%
with open('pretrain_froc_values', 'w+') as f:
    for i in tqdm(range(10)):
        num_classes = 2  # NODE21 classes + 1 background
        model = get_model(num_classes, True)
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)    

        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, targets, img_name in loaders['train']:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backprop
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                running_loss += losses.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(loaders['train'])}")

        auc, froc_sens  = compute_froc(model, loaders['test'])
        print(f"The FROC value for iteration {i} is {auc}")
        print(f"The FROC sensitivity for iteration {i} is {froc_sens}")
        f.write(f"The FROC auc value  and sensitivities for iteration {i} is {auc} and {froc_sens}\n")

