import onnx
import onnx_plaidml.backend
import numpy as np
import time

model = onnx.load('onnx_models/candy_256x256_zeropad.onnx')

data = np.arange(0.0,3.0*256.0*256.0,1.0).reshape([1,3,256,256])

rep = onnx_plaidml.backend.prepare(model)

for i in range (500):
    t0 = time.time()
    #output = onnx_plaidml.backend.run_model(model, [data])
    output = rep.run([data])
    t1 = time.time()

    #print(output)
    print("inference time:{}".format(t1-t0))


