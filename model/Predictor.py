from Generator import Generator
import torch
import cv2
import numpy as np

class Predictor():
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        gen = Generator(dataset='Hayao')
        checkpoint = torch.load(model_path,  map_location=self.device) 
        gen.load_state_dict(checkpoint['model_state_dict'], strict=True)
        gen.eval()
        # gen.to(self.device)
        return gen

    def normalize_input(self, images):
        '''[0, 255] -> [-1, 1]'''
        return images / 127.5 - 1.0


    def denormalize_input(self, images, dtype=None):
        '''[-1, 1] -> [0, 255]'''
        images = images * 127.5 + 127.5
        if dtype is not None:
            if isinstance(images, torch.Tensor):
                images = images.type(dtype)
            else:
                images = images.astype(dtype)
        return images
    
    def input_transform(self, image_data):
        image = image_data[:,:,::-1]
        image = cv2.resize(image, (image.shape[1] // 2*2, image.shape[0] // 2*2))
        image = image.astype(np.float32)
        image = self.normalize_input(image)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)
    
    def generate(self, image_data):
        image_tensor = self.input_transform(image_data)[None,:,:,:]

        # Thực hiện dự đoán
        with torch.no_grad():
            fake_img = self.model(image_tensor)
            fake_img = fake_img.detach().cpu().numpy()
            fake_img = fake_img.transpose(0, 2, 3, 1)
            fake_img = self.denormalize_input(fake_img, dtype=np.int16)

        return fake_img[0][..., ::-1]
    