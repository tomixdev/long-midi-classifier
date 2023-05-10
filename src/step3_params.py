import torchvision.models 
from torchvision.transforms.functional import InterpolationMode
import torch
import params

RAND_SEED = params.RAND_SEED

DATA_DIR = "../03-generated-midi-image-set01/bw_piano_roll_img_with_pitch_as_row"
# DATA_DIR = "../03-generated-midi-image-set01/bw_piano_roll_img_with_velocity_as_row"
RESULT_IMG_DIR = "../05-visualization/"

pretraine_model = torchvision.models.resnet18(pretrained=True)
resized_img_size = (224, 224)
interpolation_method = InterpolationMode.NEAREST

batch_size = 10
n_epochs = 10
lr = 0.001

criterion = torch.nn.CrossEntropyLoss()
optim_sgd_momentum = 0.9
