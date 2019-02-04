

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

var imgUrl_128 = "./images/amber_128x128.jpg";
var canvas_128_src = "canvas_128_src";
var canvas_128_dst = "canvas_128_dst";

var imgUrl_256 = "./images/amber_256x256.jpg";
var canvas_256_src = "canvas_256_src";
var canvas_256_dst = "canvas_256_dst";

window.onload = function() {
  loadImage(imgUrl_128, canvas_128_src);
  loadImage(imgUrl_256, canvas_256_src);
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

function contextToTensor (canvasId) {
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

function tensorToContext (tensor, canvasId) {
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

//
// styleTransfer()
//     sess - ONNX.js session (from an .onnx)
//     srcCanvasId - source canvas ID
//     dstCanvasId - dest canvas ID
//
async function styleTransfer(sess, onnxOutputId, srcCanvasId, dstCanvasId) {

  inputTensor = contextToTensor(srcCanvasId);
  
  // run inference
  const t0 = performance.now();
  const pred = await sess.run([inputTensor]);
  const t1 = performance.now();
  
  // get the result and set it to dest context
  const output = pred.get(onnxOutputId);
  tensorToContext (output, dstCanvasId);

  return t1-t0;
}

var sess = null;

var infer_128 = {
  modelUrl: "./onnx_models/candy_128x128.onnx",
  outputNodeName: "433",
  canvas_src: canvas_128_src,
  canvas_dst: canvas_128_dst,
  output_html_id: "predictions_128"
};

var infer_256 ={
  modelUrl: "./onnx_models/candy_256x256.onnx",
  outputNodeName: "433",
  canvas_src: canvas_256_src,
  canvas_dst: canvas_256_dst,
  output_html_id: "predictions_256"
};

var infer = infer_128;

var counter = 0;
async function runExample() {
  counter =0;
  runInference();
}

async function runInference(){
  setCanvasRGB(infer.canvas_dst, 0xFF, 0, 0);
  //setCanvasRGB(canvas_256_dst, 0, 0xFF, 0);

  // load model and create session
  if (sess == null) {
    sess = new onnx.InferenceSession();
    await sess.loadModel(infer.modelUrl);
  }

  // run inference
  inferTime = await styleTransfer (sess, infer.outputNodeName, infer.canvas_src, infer.canvas_dst);
  outputHtml = document.getElementById(infer.output_html_id);
  outputHtml.innerHTML += "<p>inference time: " + inferTime + "ms.</p>"; 

  if (counter++ != 10){
    setTimeout(()=>{runInference();},1000)
  }
}
