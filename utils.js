

const srcImageBaseUrl  = "./images/amber_###x###.jpg";
const onnxModelBaseUrl = "./onnx_models/candy_###x###.onnx";
const onnxOutputNodeName = "433";

const srcCanvasId = "canvas_src";
const dstCanvasId = "canvas_dst";

const totalRunCount = 3;
const inferTimeOut  = 1000; // in ms

// html utilities
function onSizeSelectChange() {
  var sizeStr = document.getElementById("img_size").value;
  
  generateInferStyleHTML(sizeStr);
}

function generateInferStyleHTML(sizeStr) {
  // generate HTML
  var html = "";

  html += "<canvas id='" + srcCanvasId + "' height=" + sizeStr + " width=" + sizeStr + " ></canvas>";
  html += "<img src='images/candy.jpg' height=" + sizeStr + " />";
  html += "<canvas id='" + dstCanvasId + "' height=" + sizeStr + " width=" + sizeStr + " ></canvas>";

  inferStyle.innerHTML = html;

  // load source image
  imgUrl = srcImageBaseUrl.replace(/###/g,sizeStr);

  loadImage (imgUrl, srcCanvasId);

  // fill dest canvas to green
  setCanvasRGB (dstCanvasId, 0x00, 0xFF, 0x00);
}

window.onload = function() {
  var sizeStr = document.getElementById("img_size").value;
  
  generateInferStyleHTML(sizeStr);
}

//
// loadImage()
//     imageUrl - URL for image to load
//     canvasId - target canvas ID
//     completeProc - callback when complete
//
function loadImage (imageUrl, canvasId, completeProc=null) {
  var img = new Image();
  img.src = imageUrl
  img.onload=function() {
    var ctx = document.getElementById(canvasId).getContext("2d");
    ctx.drawImage(img, 0, 0);

    if (completeProc != null) {
      completeProc();
    }
  }
}


// canvas utilities
function setCanvasRGB(canvasId, r, g, b, a=0xFF){
  var ctx = document.getElementById(canvasId).getContext("2d");
  var h = ctx.canvas.height;
  var w = ctx.canvas.width;

  var imgData = ctx.getImageData(0, 0, w, h);
  var data = imgData.data;

  i=0;
  for (var y=0; y<h; y++) {
    for (var x=0; x<w; x++) {
      data[i++]=r;
      data[i++]=g;
      data[i++]=b;
      data[i++]=a;
    }
  }

  ctx.putImageData(imgData, 0, 0);

}

function canvasToTensor (canvasId) {
  var ctx = document.getElementById(canvasId).getContext("2d");

  const n = 1
  const c = 3
  const h = ctx.canvas.height
  const w = ctx.canvas.width
  
  // load src context to a tensor
  var srcImgData = ctx.getImageData(0, 0, w, h);
  var src_data = srcImgData.data;
  
  const out_data   = new Float32Array(n*c*h*w);
  
  var src_idx = 0;
  var out_idx_r = 0;
  var out_idx_g = out_idx_r + h*w;
  var out_idx_b = out_idx_g + h*w;

  for (var y=0; y<h; y++) {
    for (var x=0; x<w; x++) {
      src_r = src_data[src_idx++];
      src_g = src_data[src_idx++];
      src_b = src_data[src_idx++];
      src_idx++;

      out_data[out_idx_r++] = src_r;// / 255.0;// * 255.0;
      out_data[out_idx_g++] = src_g;// / 255.0;//128.0 * 255.0;
      out_data[out_idx_b++] = src_b;// / 255.0;//0 * 255.0;
    }
  }

  const out  = new onnx.Tensor(out_data, 'float32', [n,c,h,w]);

  return out;
}

function tensorToCanvas (tensor, canvasId) {
  const h = tensor.dims[2];
  const w = tensor.dims[3];
  var t_data = tensor.data;

  t_idx_r = 0;
  t_idx_g = t_idx_r + (h*w);//(h*2*w*2);
  t_idx_b = t_idx_g + (h*w);//(h*2*w*2);

  var dst_ctx = document.getElementById(canvasId).getContext("2d");
  var dst_ctx_imgData = dst_ctx.getImageData(0, 0, w, h);
  var dst_ctx_data = dst_ctx_imgData.data;
  
  dst_idx = 0;
  for (var y=0; y<h; y++) {
    for (var x=0; x<w; x++) {
      r = t_data[t_idx_r++];
      g = t_data[t_idx_g++];
      b = t_data[t_idx_b++];

      dst_ctx_data[dst_idx++]=r;
      dst_ctx_data[dst_idx++]=g;
      dst_ctx_data[dst_idx++]=b;
      dst_ctx_data[dst_idx++]=0xFF;
    }
  }
  
  dst_ctx.putImageData(dst_ctx_imgData, 0, 0);
}

function runFNSCount(onnxSess, counter, timeOut, inferCompleteCallback, runCompleteCallback){
  setCanvasRGB(dstCanvasId, 0xFF, 0, 0);  // canvas context updating only works in event loop...
  //inferResults.innerHTML += "1 ";

  // run inference
  inputTensor = canvasToTensor(srcCanvasId);

  const inferT0 = performance.now();
  onnxSess.run([inputTensor]).then((pred)=>{
    const inferT1 = performance.now();

    // get the result and callback complete function
    const output = pred.get(onnxOutputNodeName);
    const inferTime = inferT1 - inferT0;

    inferCompleteCallback (output, inferTime);

    counter--;
    if (counter == 0){
      runCompleteCallback();
    } else {
      setTimeout(()=>{runFNSCount(onnxSess, counter, timeOut, inferCompleteCallback, runCompleteCallback);}, timeOut)
    }
  });
}

var inferTimeList = [];
function FNSInferCompleteCallback (output, inferTime) {
  //if ((inferTimeList.length % 2) == 0) {
    tensorToCanvas (output, dstCanvasId);
  //}

  inferResults.innerHTML += "inference time: " + inferTime + "<br/>";
  inferTimeList.push(inferTime);
}

function FNSRunCompleteCallback() {
  const len = inferTimeList.length;
  var total = 0;
  for (var i=1; i<len; i++) {
    total += inferTimeList[i];
  }

  const m = total / (len - 1);
  
  inferResults.innerHTML += "average inference time (excluding 1st inference): " + m + "<br/>";
}

function onRunFNSInfer() {
  setCanvasRGB (dstCanvasId, 0xFF, 0, 0);

  var sizeStr = document.getElementById("img_size").value;
  var onnxModelUrl = onnxModelBaseUrl.replace(/###/g,sizeStr);

  onnxSess = new onnx.InferenceSession();
  
  inferResults.innerHTML += "loading " + onnxModelUrl + "<br/>";
  
  const loadModelT0 = performance.now();
  onnxSess.loadModel(onnxModelUrl).then(()=>{
    const loadModelT1 = performance.now();
    inferResults.innerHTML += "load time: " + (loadModelT1 - loadModelT0) + "<br/>";

    inferTimeList = [];
    runFNSCount(onnxSess, totalRunCount, inferTimeOut, FNSInferCompleteCallback, FNSRunCompleteCallback);
  });
}