

const srcImageBaseUrl  = "./images/amber_###x###.jpg";       // ### denotes the size of the image
const onnxModelThumbUrl= "./images/candy.jpg";
const onnxModelBaseUrl = "./onnx_models/candy_###x###.onnx"; // ### denotes different onnx models, corresponding to different image sizes
const onnxOutputNodeName = "433";

const srcCanvasId = "canvas_src"; // loads srcImage
const dstCanvasId = "canvas_dst"; // outputs inference output

const totalInferCount    = 30;    // total number of inferences to run.  (should be > 1, as 1st inference run always takes longer for building up the backend kernels.)
const inferDisplayTime  = 100;  // in ms, time to show the inference output.
const asyncTimeout     = 100;

const floatRounded = 4;     // number of decimal digits to show

// output canvas color during inference
const dstCanvas_r = 0xFF;
const dstCanvas_g = 0x00;
const dstCanvas_b = 0x00;

// inference time array
var inferTimeList = [];

const newLine = String.fromCharCode(13, 10);

// html events
window.onload = function() {
  var sizeStr = document.getElementById("imgSizeSelect").value;
  
  generateInferStyleHTML(sizeStr);
}

function onSizeSelectChange() {
  var sizeStr = document.getElementById("imgSizeSelect").value;
  
  generateInferStyleHTML(sizeStr);
}

var onnxSess = null;
//var counter = 0;

function onRunFNSInfer() {
  var sizeStr = document.getElementById("imgSizeSelect").value;
  var onnxModelUrl = onnxModelBaseUrl.replace(/###/g,sizeStr);

  onnxSess = new onnx.InferenceSession();
  
  // disable UI
  imgSizeSelect.disabled = true;
  runInferButton.disabled = true;

  // reset benchmark output
  inferTimeList = [];
  counter = totalInferCount;

  copyButtonDiv.innerHTML = "";
  
  inferResultsDiv.innerHTML = "<textarea id='inferResultsText' readonly cols=90 rows=20></textarea>";
  inferResultsText.innerHTML = "";
  inferResultsText.innerHTML += "loading " + onnxModelUrl + newLine;
  
  var loadModelT0 = performance.now();
  onnxSess.loadModel(onnxModelUrl).then( ()=>{
    var loadModelTStr = (performance.now() - loadModelT0).toFixed(floatRounded);
    inferResultsText.innerHTML += "load time: " + loadModelTStr + newLine;

    // warmup tensor
    warmTensor = canvasToTensor(srcCanvasId);

    const warmT0 = performance.now();
    onnxSess.run([warmTensor]).then( ()=> {
      const warmTStr = (performance.now() - warmT0).toFixed(floatRounded);

      inferResultsText.innerHTML += "warm up time: " + warmTStr + newLine;
      runFNSCount();

    });
  });
}

function onCopyToClipboard () {
  inferResultsText.select();
  document.execCommand("copy");
}

// html utilities
function generateInferStyleHTML(sizeStr) {
  // generate HTML
  var html = "";

  html += "<canvas id='" + srcCanvasId + "' height=" + sizeStr + " width=" + sizeStr + " ></canvas>";
  html += "<img src='" + onnxModelThumbUrl + "' height=" + sizeStr + " />";
  html += "<canvas id='" + dstCanvasId + "' height=" + sizeStr + " width=" + sizeStr + " ></canvas>";

  inferStyleDiv.innerHTML = html;

  // load source image
  imgUrl = srcImageBaseUrl.replace(/###/g,sizeStr);

  loadImage (imgUrl, srcCanvasId);

  // fill dest canvas to green
  setCanvasRGB (dstCanvasId, 0x00, 0xFF, 0x00);
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

// FNS Inference functions and callbacks
function FNSInferCompleteCallback (output, inferTime) {
  tensorToCanvas (output, dstCanvasId);

  inferTimeStr = inferTime.toFixed(floatRounded);
  inferResultsText.innerHTML += "inference time #" + (inferTimeList.length + 1) + ": " + inferTimeStr + newLine;
  inferTimeList.push(inferTime);
}

function FNSRunCompleteCallback() {
  const len = inferTimeList.length;
  var total = 0;
  for (var i=1; i<len; i++) {
    total += inferTimeList[i];
  }

  const m = total / len;
  const mStr = m.toFixed(floatRounded);
  inferResultsText.innerHTML += "average inference time: " + mStr + newLine;

  // add copy button
  copyButtonDiv.innerHTML = "<button onclick='onCopyToClipboard()'>Copy to clipboard</button>";

  imgSizeSelect.disabled = false;
  runInferButton.disabled = false;
}

// benchmark function
function runFNSCount(){
    
  setCanvasRGB(dstCanvasId, dstCanvas_r, dstCanvas_g, dstCanvas_b);  // clear the output canvas

  setTimeout ( () => {
    // run inference
    inputTensor = canvasToTensor(srcCanvasId);

    const inferT0 = performance.now();

    onnxSess.run([inputTensor]).then((pred)=>{
      const inferT1 = performance.now();

      // get the result and callback complete function
      const output = pred.get(onnxOutputNodeName);
      const inferTime = inferT1 - inferT0;

      //inferCompleteCallback (output, inferTime);
      FNSInferCompleteCallback(output, inferTime);

      counter--;
      if (counter == 0){
        //runCompleteCallback();
        FNSRunCompleteCallback();
      } else {
        // give the output canvas some time (dispalyTime) to display
        setTimeout( ()=>{
          //runFNSCount(onnxSess, counter, dispalyTime, inferCompleteCallback, runCompleteCallback);
          runFNSCount();
        }, inferDisplayTime);
      }
    });
  }, asyncTimeout);
}