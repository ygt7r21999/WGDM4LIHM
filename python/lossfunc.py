import torch
import torchvision.models

class PerceptualLoss(torch.nn.modules.loss._Loss):

    def __init__(self, pixel_loss=1.0, l1_loss=False, style_loss=0.0, lambda_feat=1, include_vgg_layers=('1', '2', '3', '4', '5')):
        super(PerceptualLoss, self).__init__(True, True)

        # download pretrained vgg19 if necessary and instantiate it
        vgg19 = torchvision.models.vgg.vgg19(pretrained=True)
        self.vgg_layers = vgg19.features

        # the vgg feature layers we want to use for the perceptual loss
        self.layer_name_mapping = {
        }
        if '1' in include_vgg_layers:
            self.layer_name_mapping['3'] = "conv1_2"
        if '2' in include_vgg_layers:
            self.layer_name_mapping['8'] = "conv2_2"
        if '3' in include_vgg_layers:
            self.layer_name_mapping['13'] = "conv3_2"
        if '4' in include_vgg_layers:
            self.layer_name_mapping['22'] = "conv4_2"
        if '5' in include_vgg_layers:
            self.layer_name_mapping['31'] = "conv5_2"

        # weights for pixel loss and style loss (feature loss assumed 1.0)
        self.pixel_loss = pixel_loss
        self.l1_loss = l1_loss
        self.lambda_feat = lambda_feat
        self.style_loss = style_loss

    def forward(self, input, target):

        lossValue = torch.tensor(0.0).to(input.device)
        l2_loss_func = lambda ipt, tgt: torch.sum(torch.pow(ipt - tgt, 2))  # amplitude to intensity
        l1_loss_func = lambda ipt, tgt: torch.sum(torch.abs(ipt - tgt))  # amplitude to intensity

        # get size
        s = input.size()

        # number of tensors in this mini batch
        num_images = s[0]

        # L2 loss  (L1 originally)
        if self.l1_loss:
            scale = s[1] * s[2] * s[3]
            lossValue += l1_loss_func(input, target) * (2 * self.pixel_loss / scale)
            loss_func = l2_loss_func
        elif self.pixel_loss:
            scale = s[1] * s[2] * s[3]
            lossValue += l2_loss_func(input, target) * (2 * self.pixel_loss / scale)
            loss_func = l2_loss_func

        # stack input and output so we can feed-forward it through vgg19
        x = torch.cat((input, target), 0)

        for name, module in self.vgg_layers._modules.items():

            # run x through current module
            x = module(x)
            s = x.size()

            # scale factor
            scale = s[1] * s[2] * s[3]

            if name in self.layer_name_mapping:
                a, b = torch.split(x, num_images, 0)
                lossValue += self.lambda_feat * loss_func(a, b) / scale

                # Gram matrix for style loss
                if self.style_loss:
                    A = a.reshape(num_images, s[1], -1)
                    B = b.reshape(num_images, s[1], -1).detach()

                    G_A = A @ torch.transpose(A, 1, 2)
                    del A
                    G_B = B @ torch.transpose(B, 1, 2)
                    del B

                    lossValue += loss_func(G_A, G_B) * (self.style_loss / scale)

        return lossValue

def tv_loss(x, beta=0.5):
    '''Calculates TV loss for an image `x`.
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)

    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))

def ctv_loss(xr, xi, beta=0.5):
    dh = torch.pow(xr[:, :, :, 1:] - xr[:, :, :, :-1], 2) + torch.pow(xi[:, :, :, 1:] - xi[:, :, :, :-1], 2)
    dw = torch.pow(xr[:, :, 1:, :] - xr[:, :, :-1, :], 2) + torch.pow(xi[:, :, 1:, :] - xi[:, :, :-1, :], 2)

    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))

def losscompute(Amp_i_gt, Ui_pred, Uo_pred_r, Uo_pred_i, losstype, LPIPSLoss):
    if losstype == 'L2':
        loss_l2 = torch.norm(Amp_i_gt - torch.abs(Ui_pred)) ** 2
        loss = loss_l2
    elif losstype == 'LPIPS':
        """tensor([1,1,m,n])——>tensor([1,3,m,n])"""
        # LPIPSLoss = PerceptualLoss().to(device)
        Amp_i_gt3 = Amp_i_gt.squeeze().squeeze()
        Amp_i_gt3 = Amp_i_gt3.repeat(1, 3, 1, 1)
        abs_Ui_pred3 = torch.abs(Ui_pred).squeeze().squeeze()
        abs_Ui_pred3 = abs_Ui_pred3.repeat(1, 3, 1, 1)
        loss_LPIPS = LPIPSLoss(Amp_i_gt3, abs_Ui_pred3)
        loss = loss_LPIPS
    elif losstype == 'L2_tv':
        loss_l2 = torch.norm(Amp_i_gt - torch.abs(Ui_pred)) ** 2

        wa = 1e-2
        wp = wa
        TV_A = tv_loss((Uo_pred_r ** 2 + Uo_pred_i ** 2) ** 0.5)
        TV_P = tv_loss(torch.atan2(Uo_pred_i, Uo_pred_r))
        loss_tv = wa * TV_A + wp * TV_P
        loss = loss_l2 + loss_tv
    elif losstype == 'L2_ctv':
        loss_l2 = torch.norm(Amp_i_gt - torch.abs(Ui_pred)) ** 2

        wc = 1e-2
        TV_C = ctv_loss(Uo_pred_r, Uo_pred_i)
        loss_ctv = wc * TV_C
        loss = loss_l2 + loss_ctv
    elif losstype == 'LPIPS_ctv':
        # LPIPSLoss = PerceptualLoss().to(device)
        Amp_i_gt3 = Amp_i_gt.squeeze().squeeze()
        Amp_i_gt3 = Amp_i_gt3.repeat(1, 3, 1, 1)
        abs_Ui_pred3 = torch.abs(Ui_pred).squeeze().squeeze()
        abs_Ui_pred3 = abs_Ui_pred3.repeat(1, 3, 1, 1)
        loss_LPIPS = LPIPSLoss(Amp_i_gt3, abs_Ui_pred3)

        wc = 5e-5
        TV_C = ctv_loss(Uo_pred_r, Uo_pred_i)
        loss_ctv = wc * TV_C
        loss = loss_LPIPS + loss_ctv
    return loss