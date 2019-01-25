import onnx
import onnx_plaidml.backend
import numpy as np

model = onnx.load('onnx_models/candy_128x128_zeropad.onnx')

data = np.arange(0.0,3.0*128.0*128.0,1.0).reshape([1,3,128,128])

output = onnx_plaidml.backend.run_model(model, [data])
print(output)
