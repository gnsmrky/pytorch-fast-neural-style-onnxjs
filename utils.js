
//const srcImageBaseUrl    = "./images/amber_###x###.jpg";       // ### denotes the size of the image

// model list
// mosaic - webgl - nc8
const style_mosaic_nc8_128x128 = {
  style_name: "mosaic 128x128 (nc8)",
  content_url: "./images/amber_128x128.jpg",
  width: 128,
  height: 128,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc8_128x128_onnxjs014.onnx"
};

const style_mosaic_nc8_256x256 = {
  style_name: "mosaic 256x256 (nc8)",
  content_url: "./images/amber_256x256.jpg",
  width: 256,
  height: 256,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc8_256x256_onnxjs014.onnx"
};

// mosaic - webgl - nc16
const style_mosaic_nc16_128x128 = {
  style_name: "mosaic 128x128 (nc16)",
  content_url: "./images/amber_128x128.jpg",
  width: 128,
  height: 128,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc16_128x128_onnxjs014.onnx"
};

const style_mosaic_nc16_256x256 = {
  style_name: "mosaic 256x256 (nc16)",
  content_url: "./images/amber_256x256.jpg",
  width: 256,
  height: 256,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc16_256x256_onnxjs014.onnx"
};


// mosaic - cpu - nc8
const style_mosaic_nc8_128x128_cpu = {
  style_name: "mosaic 128x128 (nc8)",
  content_url: "./images/amber_128x128.jpg",
  width: 128,
  height: 128,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc8_128x128_onnxjs014_cpu.onnx"
};

const style_mosaic_nc8_256x256_cpu = {
  style_name: "mosaic 256x256 (nc8)",
  content_url: "./images/amber_256x256.jpg",
  width: 256,
  height: 256,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc8_256x256_onnxjs014_cpu.onnx"
};

/*
// candy
const style_candy_nc8_128x128 = {
  style_name: "candy 128x128 (nc8)",
  content_url: "./images/amber_128x128.jpg",
  width: 128,
  height: 128,
  thumb_url: "./images/candy.jpg",
  model_url: "./onnx_models/candy_nc8_128x128_onnxjs014.onnx"
};

const style_candy_nc8_256x256 = {
  style_name: "candy 256x256 (nc8)",
  content_url: "./images/amber_256x256.jpg",
  width: 256,
  height: 256,
  thumb_url: "./images/candy.jpg",
  model_url: "./onnx_models/candy_nc8_256x256_onnxjs014.onnx"
};
*/

const style_list_webgl = [
  style_mosaic_nc8_128x128,
  style_mosaic_nc8_256x256,
  style_mosaic_nc16_128x128,
  style_mosaic_nc16_256x256
];

const style_list_cpu = [
  style_mosaic_nc8_128x128_cpu,
  style_mosaic_nc8_256x256_cpu
];

// html elements
const srcCanvasId = "canvas_src"; // shows srcImage
const dstCanvasId = "canvas_dst"; // outputs inference output

// global params/vars
var onnxSess = null;  // onnx.js session

const totalInferCount  = 10;    // total number of inferences to run.  (should be > 1, as 1st inference run always takes longer for building up the backend kernels.)
const inferDisplayTime = 50;  // in ms, time to show the inference output.
const asyncTimeout     = 10;

const floatRounded = 3;     // number of decimal digits to show

// output canvas color during inference
const dstCanvas_r = 0xFF;
const dstCanvas_g = 0x00;
const dstCanvas_b = 0x00;

// inference time array
var inferTimeList = [];
var inferCountDown = 0;

// inference result text
var inferResultStr = "";
const newLine = String.fromCharCode(13, 10);

// html events
window.onload = function() {
  htmlGenerateStyleList(style_list_webgl);
}

function htmlGenerateStyleList(list) {
  styleSelect.innerHTML = "";
  for (i=0; i<list.length; i++) {
    if (i==0) {
      styleSelect.innerHTML += "<option value='" + i + "' selected='selected'>" + list[i].style_name + "</option>";
    } else {
      styleSelect.innerHTML += "<option value='" + i + "'>" + list[i].style_name + "</option>";
    }
  }

  var styleIdx = document.getElementById("styleSelect").value;
  
  generateInferStyleHTML(styleIdx);
}

function getStyleList () {
  const backend = document.getElementById("backendSelect").value;

  if (backend == 'webgl') {
    return style_list_webgl;
  } else if (backend == 'cpu') {
    return style_list_cpu;
  } else {
    return style_list_webgl;
  }
}

function onBackendChange() {
  // re-generate style list
  const style_list = getStyleList();
  htmlGenerateStyleList(style_list);
}

function onSizeSelectChange() {
  var styleIdx = document.getElementById("styleSelect").value;
  
  generateInferStyleHTML(styleIdx);
}

function formatFloat (f) {
  return f.toFixed (floatRounded);
}

function asyncSetHtml (elemNode, html) {
  var p = new Promise( (resolve, reject) => {
    elemNode.innerHTML = html;
    setTimeout (resolve, 0);
  });

  return p;
}

function onRunFNSInfer() {
  const styleIdx = document.getElementById("styleSelect").value;
  const style = getStyleList()[styleIdx];

  //var onnxModelUrl = onnxModelBaseUrl.replace(/###/g,sizeStr);
  const onnxModelUrl = style.model_url;
  
  const backend = document.getElementById("backendSelect").value;
  //onnxSess = new onnx.InferenceSession();
  onnxSess = new onnx.InferenceSession({backendHint: backend});

  // reset benchmark output
  inferTimeList = [];
  
  inferCountDown = totalInferCount;
  
  // disable UI
  styleSelect.disabled = true;
  runInferButton.disabled = true;

  // html text area
  copyButtonDiv.innerHTML = "";
  
  inferResultsDiv.innerHTML = "<textarea id='inferResultsText' readonly cols=90 rows=20></textarea>";

  inferResultStr = "PyTorch fast-neural-style (FNS) benchmark using ONNX.js " + onnxjs_version + newLine;

  // date & time
  const currentDate = new Date();

  var date  = currentDate.getDate();
  var month = currentDate.getMonth();
  var year  = currentDate.getFullYear();
  var hour  = currentDate.getHours();
  var min   = currentDate.getMinutes();
  var sec   = currentDate.getSeconds();
  var dateStr = year + "/" + (month + 1) + "/" + date + "     " + hour + ":" + min + ":" + sec  + newLine + newLine;

  inferResultStr += "Date: " + dateStr;

  // log browser info
  var uap = new UAParser();
  uap.setUA(navigator.userAgent);
  var uapRes = uap.getResult();

  inferResultStr += "os: "      + uapRes.os.name      + " " + uapRes.os.version      + newLine;
  inferResultStr += "browser: " + uapRes.browser.name + " " + uapRes.browser.version + newLine;
  inferResultStr += "engine: "  + uapRes.engine.name  + " " + uapRes.engine.version  + newLine;

  inferResultStr += newLine;

  // log cpu arch info
  inferResultStr += "cpu arch: " + uapRes.cpu.architecture + newLine;

  // log gpu info
  var glCtx = glcanvas.getContext("webgl") || glcanvas.getContext("experimental-webgl");
  if (glCtx == null) {
    inferResultStr += "cannot get 'webgl' context..." + newLine;
  } else {
    var glInfo = glCtx.getExtension("WEBGL_debug_renderer_info");
    if (glInfo != null) {
      inferResultStr += "gpu: " + glCtx.getParameter(glInfo.UNMASKED_RENDERER_WEBGL) + newLine;
    } else {
      inferResultStr += "gpu: unknown" + newLine;
    }
  }
  inferResultStr += newLine;

  // log backend info
  inferResultStr += "ONNX.js backend: " + backend + newLine;

  // log inference info
  inferResultStr += "loading '" + onnxModelUrl + "'" + newLine;
  inferResultsText.innerHTML = inferResultStr;
  
  var loadModelT0 = performance.now();
  onnxSess.loadModel(onnxModelUrl).then( ()=>{
    var loadModelTStr = formatFloat(performance.now() - loadModelT0) + "ms";
    inferResultStr += "load time: " + loadModelTStr + newLine;

    inferResultStr += "warming up tensors... " + newLine;

    asyncSetHtml(inferResultsText, inferResultStr).then( ()=>{

      // warmup tensor
      warmTensor = canvasToTensor(srcCanvasId);
  
      const warmT0 = performance.now();
      onnxSess.run([warmTensor]).then( (output)=> {
        const warmTStr = formatFloat(performance.now() - warmT0) + "ms";
  
        inferResultStr += "warm up time: " + warmTStr + newLine;
        inferResultStr += newLine;

        asyncSetHtml(inferResultsText, inferResultStr).then( ()=>{
          runFNSCount();
        });
      });
    });
  });
}

function onCopyToClipboard () {
  inferResultsText.select();
  document.execCommand("copy");
}

// html utilities
function generateInferStyleHTML(styleIdx) {
  style = getStyleList()[styleIdx];

  // generate HTML
  var html = "";

  html += "<canvas id='" + srcCanvasId     + "' height=" + style.height + " width=" + style.width + " ></canvas>";
  html += "<img src='"   + style.thumb_url + "' height=" + style.height + " />";
  html += "<canvas id='" + dstCanvasId     + "' height=" + style.height + " width=" + style.width + " ></canvas>";

  inferStyleDiv.innerHTML = html;

  // load source image
  //imgUrl = srcImageBaseUrl.replace(/###/g,style.width);
  imgUrl = style.content_url;

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

      out_data[out_idx_r++] = src_r;
      out_data[out_idx_g++] = src_g;
      out_data[out_idx_b++] = src_b;
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
  t_idx_g = t_idx_r + (h*w);
  t_idx_b = t_idx_g + (h*w);

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
function FNSRunCompleteCallback() {
  const len = inferTimeList.length;
  var total = 0;
  for (var i=0; i<len; i++) {
    total += inferTimeList[i];
  }

  const m = total / len;
  const mStr = formatFloat(m) + "ms";

  inferResultStr += newLine;
  inferResultStr += "average inference time: " + mStr + newLine;

  inferResultsText.innerHTML = "```" + newLine + inferResultStr + "```" + newLine;
  inferResultsText.scrollTop = inferResultsText.scrollHeight; // scroll to bottom

  // add copy button
  copyButtonDiv.innerHTML = "<button onclick='onCopyToClipboard()'>Copy to clipboard</button>";

  styleSelect.disabled = false;
  runInferButton.disabled = false;
}

// benchmark function
function runFNSCount(){

  // reset output canvas color
  var p = new Promise ((resolve, reject) => {
    setCanvasRGB(dstCanvasId, dstCanvas_r, dstCanvas_g, dstCanvas_b);  // clear the output canvas
    setTimeout (resolve, 10);
  });

  p.then( ()=>{
    inputTensor = canvasToTensor(srcCanvasId);

    // run inference
    const inferT0 = performance.now();
    onnxSess.run([inputTensor]).then((pred)=>{
      const inferTime = performance.now() - inferT0;
      const inferTimeStr = formatFloat(inferTime) + "ms";

      inferTimeList.push(inferTime);

      // get the result and callback complete function
      //const output = pred.get(onnxOutputNodeName);
      const output = pred.values().next().value;  // consume output this way so no need to specify output node name.
                                                  //    only for the case of single output node.  
      
      // set output canvas
      tensorToCanvas (output, dstCanvasId);

      inferResultStr += "inference time #" + (inferTimeList.length) + ": " + inferTimeStr + newLine;
      asyncSetHtml(inferResultsText, inferResultStr).then( ()=> {
        inferResultsText.scrollTop = inferResultsText.scrollHeight; // scroll to bottom

        inferCountDown--;
        if (inferCountDown == 0){
          FNSRunCompleteCallback();
        } else {
          // give the output canvas some time (inferDisplayTime) to display
          setTimeout( ()=>{
            runFNSCount();
          }, inferDisplayTime);
        }
      });
    });
  });
}