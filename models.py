import torch
import torch.nn as nn
import torch.nn.parallel



class _netG_2(nn.Module):
    def __init__(self):
        super(_netG_2, self).__init__()
        ngf = 64

        self.main_0 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, bias=False)
        self.main_1 = nn.BatchNorm2d(512)
        self.main_2 = nn.LeakyReLU(0.2, inplace=True)

        self.main_3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.main_4 = nn.BatchNorm2d(256)
        self.main_5 = nn.LeakyReLU(0.2, inplace=True)

        self.main_6 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.main_7 = nn.BatchNorm2d(128)
        self.main_8 = nn.LeakyReLU(0.2, inplace=True)

        self.main_9 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.main_10 = nn.BatchNorm2d(64)
        self.main_11 = nn.LeakyReLU(0.2, inplace=True)

        self.main_extra_layers_0_64_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.main_extra_layers_0_64_batchnorm = nn.BatchNorm2d(64)
        self.extra_relu = nn.LeakyReLU(0.2, inplace=True)

        self.main_final_layer_deconv = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input):
        input = input.to('cuda')
        x = self.main_2(self.main_1(self.main_0(input)))
        x = self.main_5(self.main_4(self.main_3(x)))
        x = self.main_8(self.main_7(self.main_6(x)))
        x = self.main_11(self.main_10(self.main_9(x)))

        x = self.extra_relu(self.main_extra_layers_0_64_batchnorm(self.main_extra_layers_0_64_conv(x)))

        x = self.tanh(self.main_final_layer_deconv(x))

        return x


model_path = 'netG.pth'
model = _netG_2()  # ngpu, nz, nc, ngf, n_extra_layers
state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('main.', 'main_').replace('extra-layers-0.64', 'extra_layers_0_64').replace('.conv', '_conv').replace('.batchnorm', '_batchnorm').replace('final_layer.deconv', 'final_layer_deconv')
    new_state_dict[new_key] = value
model.load_state_dict(new_state_dict)
model.to('cuda')
print(model)


import os
import json
import ezkl

model_path = os.path.join('network.onnx')
compiled_model_path = os.path.join('network.ezkl')
pk_path = os.path.join('test.pk')
vk_path = os.path.join('test.vk')
settings_path = os.path.join('settings.json')

witness_path = os.path.join('witness.json')
data_path = os.path.join('input.json')
cal_data_path = os.path.join('cal_data.json')


# Create a random input
x = torch.randn(1, 100, 1, 1,requires_grad=True) # Assuming input size is (batch_size, nz, 1, 1)

model.to('cuda')
# Flips the neural net into inference mode
model.eval()

# Export the model to ONNX
torch.onnx.export(model,
                  x,
                  model_path,
                  verbose=True,
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

# Prepare input data
data_array = (x.detach().numpy()).reshape([-1]).tolist()

# Save input data to a JSON file
data = {'input_data': [data_array]}
with open('input.json', 'w') as json_file:
    json.dump(data, json_file)