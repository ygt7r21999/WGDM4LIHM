import torch
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

def readdata(dataname):
    data_np = imread(dataname)
    if data_np.ndim > 2:
        data_np = data_np[:,:,0]
    data_ts = torch.tensor(data_np)
    data_ts = data_ts.float()
    return data_ts

def showdata(data):
    data = data.detach().cpu().numpy()
    plt.imshow(data,cmap="gray")
    plt.show()

def ASM_Prop(U_in, psize, dist, lamda):
    # coordinate definition
    [Y, X] = U_in.size()
    Lx = X * psize
    Ly = Y * psize
    [nx, ny] = torch.meshgrid(torch.linspace(-X / 2 + 1,X / 2, X),\
                              torch.linspace(-Y / 2 + 1,Y / 2, Y))
    fx = nx / Lx
    fy = ny / Ly

    k = 2 * np.pi / lamda
    inside_sqrt = 1 - (lamda * fx)**2 - (lamda * fy)**2
    inside_sqrt[inside_sqrt < 0] = 0
    Transfer_Matrix = torch.exp(1j * k * dist * torch.sqrt(inside_sqrt))
    Transfer_Matrix = torch.fft.ifftshift(Transfer_Matrix)

    # # U_in重构
    # U_in_r = torch.real(U_in)
    # U_in_i = torch.imag(U_in)
    # U_in = torch.stack((U_in_r,U_in_i), dim=2)
    # U_in = torch.view_as_complex(U_in)

    FU_in = torch.fft.fft2(U_in)
    FU_out = FU_in * Transfer_Matrix
    U_out = torch.fft.ifft2(FU_out)

    return U_out

def ASM_Prop_gpu(U_in_r, U_in_i, psize, dist, lamda, device):
    # coordinate definition
    if U_in_r.ndim == 4:
        [_, _, Y, X] = U_in_r.size()
    elif U_in_r.ndim == 5:
        [_, _, Y, X, D] = U_in_r.size()

    Lx = X * psize
    Ly = Y * psize
    [nx, ny] = torch.meshgrid(torch.linspace(-X / 2 + 1,X / 2, X),\
                              torch.linspace(-Y / 2 + 1,Y / 2, Y))
    fx = nx / Lx
    fy = ny / Ly

    k = 2 * np.pi / lamda
    inside_sqrt = 1 - (lamda * fx)**2 - (lamda * fy)**2
    inside_sqrt[inside_sqrt < 0] = 0

    if U_in_r.ndim == 5:
        Transfer_Matrix = torch.zeros([Y,X,D]).to(device)
        Transfer_Matrix[:, :, 0] = torch.exp(1j * k * dist * torch.sqrt(inside_sqrt)).to(device)
        Transfer_Matrix[:, :, 1] = torch.exp(1j * k * dist * torch.sqrt(inside_sqrt)).to(device)
        Transfer_Matrix[:, :, 2] = torch.exp(1j * k * dist * torch.sqrt(inside_sqrt)).to(device)
    else:
        Transfer_Matrix = torch.exp(1j * k * dist * torch.sqrt(inside_sqrt)).to(device)

    Transfer_Matrix = torch.fft.ifftshift(Transfer_Matrix)

    U_in = U_in_r + 1j*U_in_i
    FU_in = torch.fft.fft2(U_in)
    FU_out = FU_in * Transfer_Matrix
    U_out = torch.fft.ifft2(FU_out)

    return U_out

def main():
    Amp_o_gt = torch.sqrt(readdata("I0.bmp"))
    Pha_o_gt = readdata("P0.bmp")
    showdata(Amp_o_gt)
    showdata(Pha_o_gt)
    Uo_gt = Amp_o_gt*torch.exp(1j*Pha_o_gt)
    psize = 1.34e-3 # unit:mm
    dist = 1.21
    lamda = 0.534e-3
    Ui_gt = ASM_Prop(Uo_gt, psize, dist, lamda)
    showdata(torch.abs(Ui_gt))

if __name__ == '__main__':
    main()