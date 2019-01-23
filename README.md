## Run PyTorch exported .onnx models from [Fast-Neural-Style](https://github.com/pytorch/examples/tree/master/fast_neural_style) in web browser

This project is to give anyone who would like to run PyTorch's Fast-Neural-Style example in web browsers.  The performance is by no means optimized.  I used it as an example to learn what it takes to make the PyTorch codes portable to other deep learning frameworks, such as [ONNX.js](https://github.com/Microsoft/onnxjs).

Sources from the following:
- ONNX model files exported from [PyTorch fast-neural-style example](https://github.com/pytorch/examples/tree/master/fast_neural_style).
- [ONNX.js add example](https://github.com/Microsoft/onnxjs/tree/master/examples/browser/add).

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

