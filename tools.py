import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import time
import os
from PIL import Image
from torchvision import transforms
from UNet import UNet
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bce = nn.BCEWithLogitsLoss()
def BCELoss(preds, targets):
    ce_loss = bce(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss,  acc
def get_data_split(trainset_path, valset_path):
    f = open(trainset_path, "r")
    X_train = []
    for x in f:
        X_train.append((x+'.jpg').replace('\n',''))
    f.close()
    f = open(valset_path, "r")
    X_val = []
    for x in f:
        X_val.append((x+'.jpg').replace('\n',''))
    f.close()
    return X_train, X_val
def get_mean_std(loader):
    num_pixels = 0
    num_images = 1
    for bx, data in enumerate(loader):
        if(bx == 0):
            images = data
            img_mean, img_std = images.mean([2,3]), images.std([2,3])
            batch_size, num_channels, height, width = images.shape
            num_pixels += height * width
            mean = img_mean
            std = img_std
        else:
            images = data
            img_mean, img_std = images.mean([2,3]), images.std([2,3])
            batch_size, num_channels, height, width = images.shape
            num_pixels += height * width
            mean += img_mean
            std += img_std
            num_images += 1
    mean = mean[0]
    std = std[0]
    for i in range(3):
        mean[i] = mean[i]/num_images
        std[i] = std[i]/num_images
    return mean, std
def label_img(img, model, img_shape, DEVICE):
    model.eval()
    image = img.to(device=DEVICE)
    image = image[None]
    _mask = model.pred(image)
    _mask = torch.sigmoid(_mask)
    _mask = _mask.squeeze(1).permute(1,2,0).to("cpu")
    _mask = _mask.detach().numpy()
    _mask[_mask >= 0.5] = 1.0
    _mask[_mask < 0.5] = 0.0
    _mask = cv2.resize(_mask, (img_shape[1],img_shape[0]))
    return _mask
def blend_mask(img, pred):
    blended = np.copy(img)
    idxs = np.where(pred == 1.0)
    for i in range(len(idxs[0])):
        blended[idxs[0][i]][idxs[1][i]] = [255,0,255]
    return blended
def generate_mask(img, pred):
	blended = np.zeros(shape=(img.shape[0],img.shape[1]), dtype=int)
	idxs = np.where(pred == 1.0)
	for i in range(len(idxs[0])):
		blended[idxs[0][i]][idxs[1][i]] = 255
	return blended
def render_mask(img, pred):
    blended = np.zeros(img.shape, dtype=int)
    idxs = np.where(pred == 1.0)
    for i in range(len(idxs[0])):
        blended[idxs[0][i]][idxs[1][i]] = [255,0,228]
    return blended
@torch.no_grad()
def validate_batch(model, data, criterion, DEVICE):
    model.eval()
    ims, targets = data
    images = ims.to(device=DEVICE)
    masks = targets.float().unsqueeze(1).to(device=DEVICE)
    _masks = model(images)
    loss, acc = criterion(_masks, masks)
    return loss
def create_masks(img_dl, model, img_shape, source_dir, destination_dir, DEVICE):
	times = []
	accumulated = [0.0]
	for bx, data in enumerate(img_dl):
		images, names = data
		start = time.time()
		pred = label_img(images.__getitem__(0), model, img_shape, DEVICE)
		end = time.time()
		frame_time = end - start
		times.append(frame_time)
		accumulated.append(accumulated[len(accumulated) - 1] + frame_time)

		for i in range(len(names)):
			img_path = os.path.join(source_dir, names[i])
			original_img = np.array(Image.open(img_path).convert("RGB"))
			resized = cv2.resize(original_img, (img_shape[1],img_shape[0]), interpolation = cv2.INTER_AREA)
			blended = blend_mask(resized, pred)
			mask = Image.fromarray(blended)
			mask.save(destination_dir + names[i])
	return times, accumulated
def make_video(path, shape, VIDEO_NAME):
    FRAME_RATE = 10

    archivos = sorted(os.listdir(path))
    img_array = []
 
    for x in range (0,len(archivos)):
        nomArchivo = archivos[x]
        dirArchivo = path + str(nomArchivo)
        img = cv2.imread(dirArchivo)
        resized = cv2.resize(img, (shape[1],shape[0]), interpolation = cv2.INTER_AREA)
        img_array.append(resized)
    video = cv2.VideoWriter(VIDEO_NAME, cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, (shape[1],shape[0]))
    for i in range(0, len(img_array)):
        video.write(img_array[i])
    video.release()
def save_summary(train_loss, train_acc, val_loss, val_acc, path):
    f = open(path, "a")
    string = f'{train_loss},{train_acc},{val_loss},{val_acc}\n'
    f.write(string)
    f.close()
