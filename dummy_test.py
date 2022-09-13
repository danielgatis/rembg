from rembg import remove
from PIL import Image

file = "1657733712_379f2303-c0e3-43cf-b854-bdd834c806c8_original.png"
model_path = "u2net.onnx" #"isnet_bce_itr_181250_train_0.088775_tar_0.008633.onnx" # "u2net.onnx"

input = Image.open(file)
output = remove(input, model_name=model_path)
output.save("tmp.png")
