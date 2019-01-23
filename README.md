## Run [PyTorch fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style) in web browser + [ONNX.js](https://github.com/Microsoft/onnxjs)

This repository is to give anyone who would like to run PyTorch's fast-neural-style example in web browsers.  The performance is by no means optimized due to many workarounds for issues and limitations during the conversion process and different operator/layer support level between PyTorch and ONNX.js.  But it serves the purpose to understand what it takes to go through the entire process.

I used it as an example to practice and learn what it takes to make the PyTorch codes portable to other deep learning frameworks, such as [ONNX.js](https://github.com/Microsoft/onnxjs).

This project is based on the following open source projects:
- [PyTorch v1.0.0 - fast-neural-style example](https://github.com/pytorch/examples/tree/master/fast_neural_style) - ONNX model files.
- [ONNX.js v0.1.2 - add example](https://github.com/Microsoft/onnxjs/tree/master/examples/browser/add).

## Some story... and some notes...
It is frustrating for a deep learning beginner, like me, to go through various frameworks, model formats, model conversions, when developing and deploying a deep learning application.  Usually a deep learning framework comes with various examples.  Running such examples within the accompanied framework is usually ok.  Running examples in another framework, however, requires model conversion and the knowledge about the target framework.

The PyTorch's fast-neural-style example is the most facinating one.  Partly due to the way it is implemented provides a much finer image.  To run the inference in browser, the following 3 major steps are taken:

1. Use PyTorch to train the model (this repository uses the 4 pre-trained models.)
2. Export ONNX model files (.onnx)
3. Load the ONNX model file and run inference using ONNX.js.

These steps may seem easy, but not.  The following were the major obstacles encountered during the process:
1. Operator/layer support levels are *very different*.
   * PyTorch nn layers - [https://pytorch.org/docs/stable/nn.html](https://pytorch.org/docs/stable/nn.html)
   * PyTorch ONNX operators - [https://pytorch.org/docs/stable/onnx.html#supported-operators](https://pytorch.org/docs/stable/onnx.html#supported-operators)
   * ONNX.js operators - [https://github.com/Microsoft/onnxjs/blob/master/docs/operators.md](https://github.com/Microsoft/onnxjs/blob/master/docs/operators.md)
   * i.e. InstanceNorm2D() is supported by PyTorch, but not in ONNX export and ONNX.js.
   * (As time of writing, ONNX.js op level is the lowest among 3.)

2. Base tensor operations support levels are different.
   * i.e. Reduction operation along multiple axis, such as mean(), is supported by PyTorch and ONNX.js, but not ONNX export.

3. ONNX.js has quite a few issues.
   * i.e. pow() + mean() produces `NaN` values in javascript.

This repository shows running the ONNX.js inference.

## Install node.js
Windows:
```
https://nodejs.org/en/#download
```
Ubuntu:
```
sudo apt install nodejs npm
```

Install node.js http-server module
```
npm install http-server -g
```


## Run inference for Fast Neural Style in browser
Run node.js server locally:
```
http-server . -c-1 -p 3000
```
Open a browser and go to the following URL:
```
http://localhost:3000
```

