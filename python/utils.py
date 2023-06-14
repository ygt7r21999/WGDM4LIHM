import torch
from skimage.io import imread
import matplotlib.pyplot as plt

def normaliz(data):
    data_normliz = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    return data_normliz

"""datai:.png_file——>tensor([1,1,m,n])"""
def datai(dataname):
    data_np = imread(dataname)
    if data_np.ndim > 2:
        data_np = data_np[:, :, 0]
    data_ts = torch.tensor(data_np)
    data_ts = data_ts.unsqueeze(0).unsqueeze(0)
    data_ts = data_ts.float()
    data_ts = normaliz(data_ts)
    return data_ts

"""datao:show image of tensor([1,1,m,n])"""
def datao(data):
    data1 = data.squeeze().squeeze()
    data1 = data1.detach().cpu().numpy()
    plt.imshow(data1,cmap="gray")
    plt.show()

"""datas:tensor([1,1,m,n])——>nparray([m,n])"""
def datas(data):
    data1 = data.squeeze().squeeze()
    data1 = data1.detach().cpu().numpy()
    return data1

def main():
    img = imread("I0.bmp")
    print(img.shape)
    img = torch.from_numpy(img)
    print(img.ndim)
if __name__ == '__main__':
    main()

