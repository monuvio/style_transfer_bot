# %matplotlib inline
from PIL import Image


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


import torchvision.transforms as transforms
import torchvision.models as models

import copy


class StyleTransferModel:
    def __init__(self):
        device = 'cpu'
        imsize = 128
      
    def get_style_model_and_losses(self, style_img, content_img):
        device = 'cpu'
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
        cnn = copy.deepcopy(cnn)

        normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        normalization = Normalization(normalization_mean, normalization_std).to(device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0  
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
                
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses
      
    def transfer_style(self, content_img, style_img):
        print('Building the style transfer model..')
        device = 'cpu'
        normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
        cnn = copy.deepcopy(cnn)
        model, style_losses, content_losses = StyleTransferModel.get_style_model_and_losses(style_img, style_img, content_img)
        optimizer = StyleTransferModel.get_input_optimizer(input_img, input_img)

        print('Optimizing..')
        num_steps=100
        run = [0]
        while run[0] <= num_steps:

            def closure():
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                content_weight=1
                style_weight=100000
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)

        return input_img
    
    def get_input_optimizer(self, input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()]) 
        return optimizer
      
    def gram_matrix(self, input):
        batch_size , h, w, f_map_num = input.size() 

        features = input.view(batch_size * h, w * f_map_num) 

        G = torch.mm(features, features.t()) 


        return G.div(batch_size * h * w * f_map_num)

    def process_image(self, img_stream):
        loader = transforms.Compose([
            transforms.Resize(imsize), 
            transforms.CenterCrop(imsize),
            transforms.ToTensor()]) 

        image = Image.open(img_stream)
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            self.target = target.detach()
            self.loss = F.mse_loss(self.target, self.target )

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = StyleTransferModel.gram_matrix(target_feature, target_feature).detach()
            self.loss = F.mse_loss(self.target, self.target)# to initialize with something

        def forward(self, input):
            G = StyleTransferModel.gram_matrix(input, input)
            self.loss = F.mse_loss(G, self.target)
            return input

class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean) / self.std



