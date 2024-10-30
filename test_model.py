import os
import sys
import torch
import tqdm
import matplotlib.pyplot as plt
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"]='1'

from torch.utils.data import (DataLoader)
from model.template import DL_MUL_COVRES_DPW_FC96
from model.ghostnet import ghostnetv3
from model.efficientnet import effnetv2_s
from model.utils import TemplateDataset, load_and_cache_withlabel,PrintModelInfo,R2Score

data_type="test"
data_path_test=f"./dataset/src/{data_type}"
cached_file_test=f"./dataset/cache/{data_type}.pt"
label_path_test=f"./dataset/label/{data_type}/{data_type}.json"

BATCH_SIZE=1
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH="./output/output_model/efficent_last.ckpt"

def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,shuffle=False)  
    num_features = len(features)
    num_train = int(1.0* num_features)
    train_features = features[:num_train]
    dataset = TemplateDataset(features=train_features,num_instances=num_train)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader

def main():
    origin=[]
    pred=[]
    R2Sum=0
    #model = DL_MUL_COVRES_DPW_FC96().to(DEVICE)
    #model = ghostnetv3(width=1.0).to(DEVICE)
    model = effnetv2_s().to(DEVICE)
    ckpt = torch.load(MODEL_PATH)
    model.load_state_dict(ckpt['model'])
    PrintModelInfo(model)
    print("model load weight done.")
    dataloader_test=CreateDataloader(data_path_test,label_path_test,cached_file_test)
    test_iterator = tqdm.tqdm(dataloader_test, initial=0,desc="Iter", disable=False)
    model.eval()
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=100)
    with torch.no_grad():
        for step, (image,label) in enumerate(test_iterator):
            image,label= image.to(DEVICE),label.to(DEVICE)
            output = model(image)
            R2Sum_=R2Score(label[:,1:],output)
            R2Sum=R2Sum+R2Sum_
            origin.append(label[:,1:])
            pred.append(output)
    print("R2Sum is ",R2Sum/662)
    origin = torch.stack(origin).squeeze(1)
    pred = torch.stack(pred).squeeze(1)
    
    O2_P_origin = origin[:,0].cpu()
    CO2_P_origin = origin[:, 1].cpu()
    CH4_P_origin = origin[:, 2].cpu()
    N2_P_origin = origin[:, 3].cpu()

    O2_P_pred = pred[:, 0].cpu()
    CO2_P_pred = pred[:, 1].cpu()
    CH4_P_pred = pred[:, 2].cpu()
    N2_P_pred = pred[:, 3].cpu()
    
    # 设置绘图风格
    plt.style.use('seaborn-darkgrid')

    # 创建一个 2x2 的子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    point_size=5
    # O2_P 图
    axs[0, 0].scatter(range(len(O2_P_origin)), O2_P_origin, label='O2 (ori)', color='blue', s=point_size)
    axs[0, 0].scatter(range(len(O2_P_pred)), O2_P_pred, label='O2 (pred)', color='orange', s=point_size)
    axs[0, 0].set_title('O2 Groundtruth and Pred')
    axs[0, 0].set_xlabel('Num')
    axs[0, 0].set_ylabel('O2_P')
    axs[0, 0].legend()

    # CO2_P 图
    axs[0, 1].scatter(range(len(CO2_P_origin)), CO2_P_origin, label='CO2 (ori)', color='green', s=point_size)
    axs[0, 1].scatter(range(len(CO2_P_pred)), CO2_P_pred, label='CO2 (pred)', color='red', s=point_size)
    axs[0, 1].set_title('CO2 Groundtruth and Pred')
    axs[0, 1].set_xlabel('Num')
    axs[0, 1].set_ylabel('CO2')
    axs[0, 1].legend()

    # CH4_P 图
    axs[1, 0].scatter(range(len(CH4_P_origin)), CH4_P_origin, label='CH4 (ori)', color='purple', s=point_size)
    axs[1, 0].scatter(range(len(CH4_P_pred)), CH4_P_pred, label='CH4 (pred)', color='pink', s=point_size)
    axs[1, 0].set_title('CH4 Groundtruth and Pred')
    axs[1, 0].set_xlabel('Num')
    axs[1, 0].set_ylabel('CH4_P')
    axs[1, 0].legend()

    # N2_P 图
    axs[1, 1].scatter(range(len(N2_P_origin)), N2_P_origin, label='N2 (ori)', color='brown', s=point_size)
    axs[1, 1].scatter(range(len(N2_P_pred)), N2_P_pred, label='N2 (pred)', color='gray', s=point_size)
    axs[1, 1].set_title('N2 Groundtruth and Pred')
    axs[1, 1].set_xlabel('Num')
    axs[1, 1].set_ylabel('N2_P')
    axs[1, 1].legend()

    # 调整布局
    plt.tight_layout()

    # 保存图片，文件名可以根据需要修改
    plt.savefig('efficient.png', dpi=300, bbox_inches='tight')
    
if __name__=="__main__":
    main()