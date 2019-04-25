## Run [PyTorch fast-neural-style (FNS)](https://github.com/pytorch/examples/tree/master/fast_neural_style) in web browsers using [ONNX.js](https://github.com/Microsoft/onnxjs)

This repository is for anyone interested to run [PyTorch fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style) example in web browsers.  The _performance is by no means optimal_ due to many workarounds for issues and limitations during the conversion process and different operator/layer support level between PyTorch and ONNX.js.  But it serves the purpose to understand what it takes to go through the entire process.

It is an example for practicing and learning what it takes to make the PyTorch generated models portable to other deep learning frameworks. [ONNX.js](https://github.com/Microsoft/onnxjs) is set as the target deep learning framework as it's very new, hance still primitive.

This project is roughly based on the following open source projects:
- [PyTorch v1.0.0 - fast-neural-style example](https://github.com/pytorch/examples/tree/master/fast_neural_style) for exporting ONNX model files (.onnx).
- [ONNX.js v0.1.3 - add example](https://github.com/Microsoft/onnxjs/tree/master/examples/browser/add) for the javascript inference on the web.

#### Quick links:
This repo:
- Goto [PyTorch fast-neural-style](https://gnsmrky.github.io/pytorch-fast-neural-style-onnxjs/) for a quick working web style transfer using ONNX.js.
- Goto [PyTorch fast-neural-style web benchmark](https://gnsmrky.github.io/pytorch-fast-neural-style-onnxjs/benchmark.html) as a quick tool for performance on your browser.

How-to guides on tweaking [PyTorch fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style) repo:
- See [PyTorch fast-neural-style for web](https://github.com/gnsmrky/pytorch-fast-neural-style-for-web) for making it working with ONNX.js.
- See [Making the PyTorch to ONNX.js conversion work](https://github.com/gnsmrky/pytorch-fast-neural-style-for-web/tree/master/docs) if you are interested in more technical details.


## Simple goal - Inference on the web
The objective is simple:  
<p align="center"><b>PyTorch FNS example --> PyTorch model files (.pth) --> ONNX model files --> ONNX.js on web browsers</b></p>

There are many style transfer implementations.  PyTorch's fast-neural-style example is the most facinating one.  Partly due to the way it is implemented provides a much finer style-transfered images.  To run the inference in browser, the following 3 major steps are taken:

1. Use PyTorch to train the model (this repository uses the 4 pre-trained models.)
2. Use PyTorch's built-in ONNX export feature to export model files (.onnx)
3. Load the ONNX model files (.onnx) and run inference using ONNX.js in web browsers.

Sounds straight forward!?  Read on...

## Ugly honest/ONNX truth about the path to the web
These steps may seem easy, but in practice it is way much more complicated.  

The following were the *few of* major obstacles encountered during the process, just to give an idea on what are possible issues there may be:
1. **Operator/layer support levels are *very different*.**
   * PyTorch nn layers - [https://pytorch.org/docs/stable/nn.html](https://pytorch.org/docs/stable/nn.html)
   * PyTorch ONNX export operators - [https://pytorch.org/docs/stable/onnx.html#supported-operators](https://pytorch.org/docs/stable/onnx.html#supported-operators)
   * ONNX.js operators - [https://github.com/Microsoft/onnxjs/blob/master/docs/operators.md](https://github.com/Microsoft/onnxjs/blob/master/docs/operators.md)
   * PyTorch `nn.InstanceNorm2d()` is exported as ONNX `InstanceNormalization()`, but not supported by ONNX.js.
   * PyTorch `nn.functional.interpolate()` is exported as ONNX `Upsample()`, but not supported in ONNX.js.
   * At time of writing (Jan, 2019), PyTorch ONNX export at opset version 9 by default.  ONNX.js at ONNX opset version 7.  

2. **Base tensor opset levels are different between PyTorch, PyTorch ONNX Export and ONNX.js**
   * PyTorch ONNX export only supports reduction operation, such as `mean()`,  along 1 axis.  i.e. `torch.mean(t, [2,3])` is not supported by PyTorch ONNX export.  (Although both PyTorch and ONNX.js supports multi-axis reduction ops.)

3. **ONNX.js has quite a few issues.**
   * Same input values results in exception error.  ([ONNX.js issue #53](https://github.com/Microsoft/onnxjs/issues/53))
   * Some ops are *slow*, such as `Reshape()`, which is converted from PyTorch's `view()`.
   * `pow()` + `mean()` produces `NaN` values in javascript.
   * `pow()` op is *very* buggy.  
   ###### Note: Some were fixed in later version of ONNX.js.

4. **Dynamic tensor shapes exported by PyTorch ONNX is *very* large and hogs memory like hell.**
   * If any op node depends on input/out tensor shape dynamically when doing inferencing, the result ONNX model graph can be absurdly huge (.onnx file at ~350MB) and highly complex (Composed of multiple `Reshape` and `Gather` ops).  Although still works, it is not practical to use such model files in web browsers.

5. **ONNX.js support for Mobile devices, such as Android, is still not stable.**
   * The web site has `mosaic zero-pad` as first model as some mobile devices does not run `pad` op correctly using `webgl` backend.  All models should work correctly on desktop browsers.
   * As it can be seen, 'zero-pad' generates a lower quality stylized output.  

For more details on tweaking and working around the differences between PyTorch, Exporter and ONNX.js, please see [PyTorch fast-neural-style for web](https://github.com/gnsmrky/pytorch-fast-neural-style-for-web).

----------
# Original Repo
----------
## Run inference locally with `node.js`
ONNX.js can be served locally by `node.js` via `npm`.

Windows `npm` installer:
```
https://nodejs.org/en/#download
```
Ubuntu `npm` installation:
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

