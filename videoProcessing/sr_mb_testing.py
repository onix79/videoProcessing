from sr_model import resolve_single
from sr_model.edsr import edsr

from utils import load_image, plot_sample
from sr_model.wdsr import wdsr_b
from PIL import Image
import matplotlib
import cv2

model = edsr(scale=4, num_res_blocks=16)
model.load_weights('weights/edsr-16-x4/weights.h5')

#model = wdsr_b(scale=4, num_res_blocks=32)
#model.load_weights('weights/wdsr-b-32-x4/weights.h5')


#lr = load_image('/home/line/Desktop/bq.jpg')
lr = cv2.imread('/home/line/Desktop/bq.jpg')
sr = resolve_single(model, lr)

print('TypeOfLR_redImage:',type(lr))
print('TypeOFSR_image:',type(sr))
print('ShapeOfSR:',sr.shape)

matplotlib.image.imsave('name.jpg', sr.numpy())

#plot_sample(lr, sr)

