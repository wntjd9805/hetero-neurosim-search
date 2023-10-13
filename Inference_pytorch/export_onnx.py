import torchvision.models as models
import torch

#model= models.vgg16(pretrained=True)

model = models.vgg16(pretrained=True)
model.eval()

x = torch.randn(1,3,224,224)
torch_out = model(x)
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "vgg16.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  # training=TrainingMode.TRAINING,
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])
