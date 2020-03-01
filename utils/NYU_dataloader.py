from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms



class val_nyudataset(Dataset):
	def __init__(self, image_dir,mask_dir,classes,scale):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.images = []
		self.masks =[]
		self.images_name = os.listdir(self.image_dir)
		self.masks_name = os.listdir(self.mask_dir)
		self.scale = scale
		self.classes = classes

		for id in self.images_name:
			if 'png' in id:
				image = os.path.join(self.image_dir,str(id))
				self.images.append(image)
		for id in self.masks_name:
			if 'png' in id:
				mask = os.path.join(self.mask_dir,str(id))
				self.masks.append(mask)
		print(self.images)
		print(self.masks)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		images = self.images[index]
		masks = self.masks[index]
		image = Image.open(images)
		mask = Image.open(masks)
		original_size = image.size
		original_w ,original_h = original_size[0],original_size[1]

		if self.scale:
			modified_w,modified_h =int(original_w*self.scale),int(original_h*self.scale)
			image = transforms.Resize((modified_h,modified_w))(image)
			mask = transforms.Resize((modified_h,modified_w))(mask)
			image = np.array(image)
			image = image.transpose((2,0,1))  # transpose the  H*W*C to C*H*W
			mask = np.array(mask)

			mask = mask / (256//self.classes)
			mask = np.floor(mask)
			mask_sparse = mask
			return image, mask_sparse,images.lstrip('/home/panmeng/data/nyu_images/test_dir/')

class train_nyudataset(Dataset):
	def __init__(self,dir,scale,classes):
		self.dir = dir
		self.images = []
		self.masks = []
		self.scale = scale
		self.read(self.dir, level=5)
		self.classes = classes
		print(self.masks)
		print(self.images)
#EXTENTIONS = ['.bmp', '.JPG', '.PNG', '.jpg', '.png']

	def is_image(self,file_name):
		if 'rgb' in file_name:
				return file_name
	def is_masks(self,file_name):
		if 'sync' in file_name:
			return file_name

	def read(self,file_path, level=5):
		if level == 0:
			return None
		file_names = os.listdir(file_path)
		file_names.sort()
		for file_name in file_names:
			if self.is_image(file_name):
				self.images.append(os.path.abspath(os.path.join(file_path, file_name)))
			if self.is_masks(file_name):
				self.masks.append(os.path.abspath(os.path.join(file_path, file_name)))
			elif os.path.isdir(os.path.join(file_path, file_name)):
				self.read(os.path.join(file_path, file_name), level=level-1)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		images = self.images[index]
		masks = self.masks[index]
		image = Image.open(images)
		mask = Image.open(masks)
		original_size = image.size
		original_w ,original_h = original_size[0],original_size[1]

		if self.scale:
			modified_w,modified_h =int(original_w*self.scale),int(original_h*self.scale)
			image = transforms.Resize((modified_h,modified_w))(image)
			mask = transforms.Resize((modified_h,modified_w))(mask)
			image = np.array(image)
			image = image.transpose((2,0,1))  # transpose the  H*W*C to C*H*W
			mask = np.array(mask)
			mask = (mask / 10000 * 256)/(256//self.classes)
			mask = np.floor(mask)
			mask_sparse = mask
			return image, mask_sparse,images.lstrip('/home/panmeng/data/nyu_images/test_dir/')
