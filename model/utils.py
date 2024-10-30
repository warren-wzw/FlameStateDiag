import os
import re
import sys
import tqdm
import json
import torch
import shutil
import random
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import  Dataset
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image,ImageEnhance

"""Model info"""
def PrintModelInfo(model):
    """Print the parameter size and shape of model detail"""
    total_params = 0
    model_parments = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, param in model.named_parameters():
        num_params = torch.prod(torch.tensor(param.shape)).item() * param.element_size() / (1024 * 1024)  # 转换为MB
        print(f"{name}: {num_params:.4f} MB, Shape: {param.shape}")
        total_params += num_params
    print(f"Traing parments {model_parments/1e6}M,Model Size: {total_params:.4f} MB")     

"""Lr"""
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

"""Dataset"""   
def min_max_normalize(image):
    np_image = np.array(image).astype(np.float32)
    np_image = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))
    return torch.tensor(np_image)

def visual_result(input,filename):
    if len(input.shape)==4:
        np_image = input[0].cpu().permute(1,2,0).numpy()  # 将通道维度移到最后
    elif len(input.shape)==3:
        np_image = input.cpu().permute(1,2,0).numpy()  # 将通道维度移到最后
    if np_image.min()<0:    
        np_image = np_image * 0.5 + 0.5  # 假设图像已归一化为[-1, 1]
    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(filename)  # 在绘制图像后保存  
    
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image=transform(image)
    image = min_max_normalize(image)
    return image

def sort_key(filename):
    match = re.search(r'(\d+)_(\d+)_(\d+)\.', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    else:
        return float('inf'), float('inf'), float('inf')   
      
def load_and_cache_withlabel(image_path,label_path,cache_file,shuffle=False):
    if cache_file is not None and os.path.exists(cache_file):
        print("Loading features from cached file ", cache_file)
        features = torch.load(cache_file)
    else:
        print("Creating features from dataset at ", image_path,label_path)
        images,labels = [],[]
        for img_name in os.listdir(image_path):
            img_path = os.path.join(image_path, img_name)
            images.append(img_path)
        images=sorted(images,key=sort_key)
        with open(label_path,'r') as json_file:
             for i,line in enumerate(json_file):
                labels.append(json.loads(line))
        features = []
        def get_label_data(label):
            file=label["file:"]
            match = re.match(r'(\d+)_(\d+)', label["status"])
            if match:
                n = int(match.group(1))
                m = int(match.group(2))
                result = 7 * (n - 1) + m-1
            status=result
            O2=label["O2"]
            O2_CO2=label["O2_CO2"]
            CH4=label["CH4"]
            O2_CH4=label["CH4_CO2"]
            N2=random_number = random.uniform(0.55, 0.6)
            return file,status,O2,O2_CO2,CH4,O2_CH4,N2       
              
        total_iterations = len(images)  # 设置总的迭代次数  
        for image_path,label in tqdm.tqdm(zip(images,labels),total=total_iterations):
            processed_image=preprocess_image(image_path)
            file,status,O2,O2_CO2,CH4,O2_CH4,N2=get_label_data(label)
            O2_P=O2/(O2+O2_CO2+CH4+O2_CH4+N2)
            CO2_P=O2_CO2+O2_CH4/(O2+O2_CO2+CH4+O2_CH4+N2)
            CH4_P=CH4/(O2+O2_CO2+CH4+O2_CH4+N2)
            N2_P=N2/(O2+O2_CO2+CH4+O2_CH4+N2)
            feature = {
                "image": processed_image,
                "file": file,
                "status":status,
                "O2_P":O2_P,
                "CO2_P":CO2_P,
                "CH4_P":CH4_P,
                "N2_P":N2_P
            }
            features.append(feature)
 
        if shuffle:
            random.shuffle(features)
        if not os.path.exists(cache_file):
            print("Saving features into cached file ", cache_file)
            torch.save(features, cache_file)
    return features

class TemplateDataset(Dataset):
    def __init__(self,features,num_instances):
        self.feature=features
        self.num_instances=num_instances
    
    def __len__(self):
        return int(self.num_instances)
    
    def __getitem__(self, index):
        feature = self.feature[index]
        image=feature["image"]
        status=feature["status"]
        O2_P=feature["O2_P"]
        CO2_P=feature["CO2_P"]
        CH4_P=feature["CH4_P"]
        N2_P=feature["N2_P"]
        data_list = [status, O2_P, CO2_P, CH4_P, N2_P]
        # 将列表转化为一个tensor
        label = torch.tensor(data_list)
        
        return image,label

"""Save and load model"""
from datetime import datetime
def save_ckpt(save_path,model_name,model,epoch_index,scheduler,optimizer):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'epoch': epoch_index + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler':scheduler.state_dict()},
                        '%s%s' % (save_path,model_name))
    print("->Saving model {} at {}".format(save_path+model_name, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
"""Evaluate model"""
def CaculateAcc(output,label):
    print()

def R2Score(y_true, y_pred):
    # 计算总平方和（TSS）
    y_true_mean = torch.mean(y_true)
    total_sum_of_squares = torch.sum((y_true - y_true_mean) ** 2)
    
    # 计算残差平方和（RSS）
    residual_sum_of_squares = torch.sum((y_true - y_pred) ** 2)
    
    # 计算 R^2
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2.item()  # 返回标量值
