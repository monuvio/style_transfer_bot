from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import torch

from torchvision import transforms
import torchvision

from itertools import product
from PIL import Image
import time


class VGG(nn.Module):
	def __init__(self, pool='max'):
		super(VGG, self).__init__()
		self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

			
	def forward(self, x, out_keys):
		out = {}
		out['r11'] = F.relu(self.conv1_1(x))
		out['r12'] = F.relu(self.conv1_2(out['r11']))
		out['p1'] = self.pool1(out['r12'])
		out['r21'] = F.relu(self.conv2_1(out['p1']))
		out['r22'] = F.relu(self.conv2_2(out['r21']))
		out['p2'] = self.pool2(out['r22'])
		out['r31'] = F.relu(self.conv3_1(out['p2']))
		out['r32'] = F.relu(self.conv3_2(out['r31']))
		out['p3'] = self.pool3(out['r32'])
		out['r41'] = F.relu(self.conv4_1(out['p3']))
		out['r42'] = F.relu(self.conv4_2(out['r41']))
		out['p4'] = self.pool4(out['r42'])
		out['r51'] = F.relu(self.conv5_1(out['p4']))
		out['r52'] = F.relu(self.conv5_2(out['r51']))
		out['p5'] = self.pool5(out['r52'])
		return [out[key] for key in out_keys]

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
	def forward(self, source):
		return source

class CovarianceMatrix(nn.Module):
	def __init__(self):
		super(CovarianceMatrix, self).__init__()
	def forward(self, source):
		one, nFilter, h, w = source.size()
		m = h * w
		F = source.view(nFilter, m)
		A = torch.mean(F, dim=1).view(-1, 1)
		G = torch.mm(F, F.transpose(0, 1)).div(m) - torch.mm(A, A.transpose(0, 1))
		G.div_(nFilter)
		return G

class LayerLoss(nn.Module):
	def __init__(self, description, rawTarget, activationShift=0.0):
		super(LayerLoss, self).__init__()
		if description == 'raw':
			self.class_, self.argTpl = Identity, tuple()
		elif description == 'covariance':
			self.class_, self.argTpl = CovarianceMatrix, tuple()
		self.target = self.class_(*self.argTpl)(rawTarget).detach()

	def forward(self, source):
		out = nn.MSELoss()(
			self.class_(*self.argTpl)(source), 
			self.target,
		)
		return out

img_size = 384
prep = transforms.Compose([
	transforms.Scale(img_size),
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]), # turn to BGR
	transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]), # subtract imagenet mean
	transforms.Lambda(lambda x: x.mul_(255.0)), 
])
postpa = transforms.Compose([
	transforms.Lambda(lambda x: x.mul_(1.0/255.0)),
	transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1]), # add imagenet mean
	transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]), # turn to RGB
])
postpb = transforms.Compose([
	transforms.ToPILImage(),
])
def postp(tensor):
	return postpb(postpa(tensor).clamp(0.0, 1.0))

vgg = VGG()
vgg.load_state_dict(torch.load('/content/weights.pth'))
for param in vgg.parameters():
	param.requires_grad = False
if torch.cuda.is_available():
	vgg.cuda()

style_layers = ['r11', 'r21', 'r31', 'r41', 'r51'] 
content_layers = ['r32', 'r42']
loss_layers = style_layers + content_layers
style_weights = [1.2e3] * len(style_layers)
content_weights = [1e0] * len(content_layers)
weights = style_weights + content_weights

activationShift = 0.0

n_iter = 0

def transfer_style(content_img, style_img):
    imgs_torch = [prep(Image.open(style_img).convert('RGB')), prep(Image.open(content_img).convert('RGB'))]
    if torch.cuda.is_available():
         imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
    style_image, content_image = imgs_torch

    style_targets = [A.detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]

    opt_img = Variable(content_image.data.clone(), requires_grad=True)

    loss_fns = [
        LayerLoss('covariance', rawTarget, activationShift) for rawTarget in style_targets] + [
        LayerLoss('raw', rawTarget) for rawTarget in content_targets]
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    style_targets, content_targets = None, None

    optimizer = optim.LBFGS([opt_img])
    max_iter = 400
    show_iter = 100
    global n_iter
	
    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        global n_iter
        if n_iter % show_iter == 0:
            print('Iteration:', n_iter)
        n_iter += 1
        return loss

    while n_iter < max_iter:
        optimizer.step(closure)
		
    out_img = postp(opt_img.data[0].cpu().squeeze())
    return out_img