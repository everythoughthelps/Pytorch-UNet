from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class nyudataset(Dataset):
	def __init__(self, image_dir,mask_dir,scale):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.images = []
		self.masks =[]
		self.images_name = os.listdir(self.image_dir)
		self.masks_name = os.listdir(self.mask_dir)
		self.scale = scale

		for id in self.images_name:
			if 'png' in id:
				image = os.path.join(self.image_dir,str(id))
				self.images.append(image)
		for id in self.masks_name:
			if 'png' in id:
				mask = os.path.join(self.mask_dir,str(id))
				self.masks.append(mask)
		print(self.images_name)
		print(self.masks_name)
	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		images = self.images[index]
		masks = self.masks[index]
		image = Image.open(images)
		mask = Image.open(masks).convert('L')
		original_size = image.size
		original_w ,original_h = original_size[0],original_size[1]

		if self.scale:
			modified_w,modified_h =int(original_w*self.scale),int(original_h*self.scale)
			image = transforms.Resize((modified_h,modified_w))(image)
			mask = transforms.Resize((modified_h,modified_w))(mask)
			image = np.array(image)
			image = image.transpose((2,0,1))  # transpose the  H*W*C to C*H*W
			mask = np.array(mask)
			mask_dense = mask

			mask = mask * 0.25
			mask = np.floor(mask)
			mask_sparse = mask
			return image, mask_sparse, mask_dense