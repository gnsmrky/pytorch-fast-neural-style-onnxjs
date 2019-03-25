

// model list

// 128x128 models
// mosaic - webgl - nc8
const style_mosaic_nc8_128x128 = {
  style_name: "mosaic 128x128 (nc8)",
  width: 128,
  height: 128,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc8_128x128_onnxjs014.onnx"
};

// mosaic - webgl - nc16
const style_mosaic_nc16_128x128 = {
  style_name: "mosaic 128x128 (nc16)",
  width: 128,
  height: 128,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc16_128x128_onnxjs014.onnx"
};


// 256x256 nc8 webl models
const style_mosaic_nc8_256x256 = {
  style_name: "mosaic 256x256 (nc8)",
  width: 256,
  height: 256,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc8_256x256_onnxjs014.onnx"
};

// 256x256 nc16 webl models
// candy
const style_candy_nc16_256x256 = {
  style_name: "candy 256x256 (nc16)",
  width: 256,
  height: 256,
  thumb_url: "./images/candy.jpg",
  model_url: "./onnx_models/candy_nc16_256x256_onnxjs014.onnx"
};

// mosaic
const style_mosaic_nc16_256x256 = {
  style_name: "mosaic 256x256 (nc16)",
  width: 256,
  height: 256,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc16_256x256_onnxjs014.onnx"
};

// rain-princess
const style_rainprincess_nc16_256x256 = {
  style_name: "rain princess 256x256 (nc16)",
  width: 256,
  height: 256,
  thumb_url: "./images/rain-princess.jpg",
  model_url: "./onnx_models/rain-princess_nc16_256x256_onnxjs014.onnx"
};

// udnie
const style_udnie_nc16_256x256 = {
  style_name: "udnie 256x256 (nc16)",
  width: 256,
  height: 256,
  thumb_url: "./images/udnie.jpg",
  model_url: "./onnx_models/udnie_nc16_256x256_onnxjs014.onnx"
};


// mosaic - cpu - nc8
const style_mosaic_nc8_128x128_cpu = {
  style_name: "mosaic 128x128 (nc8)",
  width: 128,
  height: 128,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc8_128x128_onnxjs014_cpu.onnx"
};

const style_mosaic_nc8_256x256_cpu = {
  style_name: "mosaic 256x256 (nc8)",
  width: 256,
  height: 256,
  thumb_url: "./images/mosaic.jpg",
  model_url: "./onnx_models/mosaic_nc8_256x256_onnxjs014_cpu.onnx"
};

const style_list_webgl = [
  //style_mosaic_nc8_128x128,
  style_mosaic_nc8_256x256,
  //style_mosaic_nc16_128x128,
  style_mosaic_nc16_256x256,
  style_candy_nc16_256x256,
  style_rainprincess_nc16_256x256,
  style_udnie_nc16_256x256
];

const style_list_cpu = [
  style_mosaic_nc8_128x128_cpu,
  style_mosaic_nc8_256x256_cpu
];

// benchmark style list
const benchmark_style_list_webgl = [
  style_mosaic_nc8_128x128,
  style_mosaic_nc16_128x128,

  style_mosaic_nc8_256x256,
  style_mosaic_nc16_256x256,
  style_candy_nc16_256x256,
  style_rainprincess_nc16_256x256,
  style_udnie_nc16_256x256
];

const benchmark_style_list_cpu = [
  style_mosaic_nc8_128x128_cpu,
  style_mosaic_nc8_256x256_cpu
];

// content images
const content_url_list = [
  { name:"amber",     img_url:"./images/amber_256x256.jpg", credit_url: "unsplash.com" },
  { name:"boy",       img_url:"./images/boy.jpg", credit_url: "unsplash.com" },
  { name:"car",       img_url:"./images/car.jpg", credit_url: "unsplash.com" },
  { name:"cat",       img_url:"./images/cat.jpg", credit_url: "unsplash.com" },
  { name:"door",      img_url:"./images/door.jpg", credit_url: "unsplash.com" },
  { name:"flower",    img_url:"./images/flower.jpg", credit_url: "unsplash.com" },
  { name:"flower2",   img_url:"./images/flower2.jpg", credit_url: "unsplash.com" },
  { name:"lake house",img_url:"./images/lake_house.jpg", credit_url: "unsplash.com" },
  { name:"tree",      img_url:"./images/tree.jpg", credit_url: "unsplash.com" },
  { name:"urban sky", img_url:"./images/urban_sky.jpg", credit_url: "unsplash.com" },
  { name:"window",    img_url:"./images/window.jpg", credit_url: "unsplash.com" },
];

// html elements
const srcCanvasId = "canvas_src"; // shows srcImage
const dstCanvasId = "canvas_dst"; // outputs inference output

// global params/vars
var g_onnxSess = null;
const g_benchmark_contentImgUrl =  content_url_list[0].img_url;

const totalInferCount  = 10;    // total number of inferences to run.
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

    const side = (img.height < img.width) ? img.height : img.width;
    const sx = (img.width - side) / 2;
    const sy = (img.height- side) / 2;

    ctx.drawImage(img,
                  sx, sy, side, side,
                  0, 0, ctx.canvas.width, ctx.canvas.height);

    if (completeProc != null) {
      completeProc();
    }
  }
}

function isMobile() {
  var uap = new UAParser();
  uap.setUA(navigator.userAgent);

  var uapRes = uap.getResult();
  const osName = uapRes.os.name.toLowerCase();

  return (osName.indexOf("ios") >=0) || (osName.indexOf("android") >= 0);
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

function getStyleList () {
  const backendElem = document.getElementById("backendSelect");
  const backend = backendElem ? backendElem.value : 'webgl';

  if (backend == 'webgl') {
    return style_list_webgl;
  } else if (backend == 'cpu') {
    return style_list_cpu;
  } else {
    return style_list_webgl;
  }
}

function benchmark_getStyleList () {
  const backendElem = document.getElementById("backendSelect");
  const backend = backendElem ? backendElem.value : 'webgl';

  if (backend == 'webgl') {
    return benchmark_style_list_webgl;
  } else if (backend == 'cpu') {
    return benchmark_style_list_cpu;
  } else {
    return benchmark_style_list_webgl;
  }
}

function getContentList(){
  return content_url_list;
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

function onCopyToClipboard () {
  inferResultsText.select();
  document.execCommand("copy");
}


function htmlGenerateInferStyle(styleIdx, contentImgIdx){

}

// fast neural style html
function htmlGenerateUI () {
  htmlGenerateStyleList   (styleSelect);
  htmlGenerateContentList (contentImgSelect);
  htmlGenerateResult      ();
}

function htmlGenerateStyleList (styleListElem) {
  styleListElem.innerHTML = "";
  const list = getStyleList();
  for (i=0; i<list.length; i++) {
    if (i==0) {
      styleListElem.innerHTML += "<option value='" + i + "' selected='selected'>" + list[i].style_name + "</option>";
    } else {
      styleListElem.innerHTML += "<option value='" + i + "'>" + list[i].style_name + "</option>";
    }
  }

  htmlGenerateStyle(0);
}

function htmlGenerateStyle(styleIdx) {
  style = getStyleList()[styleIdx];

  // generate HTML
  var html = "";
  html += "<img src='" + style.thumb_url + "' height='" + style.height + "px' />" + "&nbsp;";

  inferStyleDiv.innerHTML = html;
}

function onStyleSelectChange() {
  // update style list and thumbnail
  const styleIdx = document.getElementById("styleSelect").value;
  htmlGenerateStyle(styleIdx);

  // update content image so size matches the style size
  const contentIdx = document.getElementById("contentImgSelect").value;
  htmlGenerateContent(contentIdx);

  // update stylized image so the size mathes the style size
  htmlGenerateResult ();

  // dispose inference session
  g_onnxSess = null;
}

function htmlGenerateContentList(contentImgList) {
  contentImgList.innerHTML = "";
  const list = getContentList();
  
  for (i=0; i<list.length; i++) {
    if (i==0) {
      contentImgList.innerHTML += "<option value='" + i + "' selected='selected'>" + list[i].name + "</option>";
    } else {
      contentImgList.innerHTML += "<option value='" + i + "'>" + list[i].name + "</option>";
    }
  }

  htmlGenerateContent(0);
}

function htmlGenerateContent(contentIdx, callback) {
  const styleIdx = document.getElementById("styleSelect").value;
  const style = getStyleList()[styleIdx];

  const content = getContentList()[contentIdx];
  
  // generate content image HTML
  var html = "";
  //html += "<img src='"   + content.img_url + "' height=" + style.height + " />" + "&nbsp;";
  html += "<canvas id='" + srcCanvasId + "' height='" + style.height + "px' width='" + style.width + "px' ></canvas>";

  contentImgDiv.innerHTML = html;
  loadImage (content.img_url, srcCanvasId, callback);

  // generate image credit
  html = "<a href='https://" + content.credit_url + "'>" + content.credit_url + "</a>";
  contentImgCredit.innerHTML = html;
}

function onContentImgSelectChange() {
  const contentIdx = document.getElementById("contentImgSelect").value;
  htmlGenerateContent(contentIdx, onRunFNSInfer);  // run style infer upon new content image is loaded
}

function htmlGenerateResult (){
  const styleIdx = document.getElementById("styleSelect").value;
  const style = getStyleList()[styleIdx];

  html = "<canvas id='" + dstCanvasId + "' height='" + style.height + "' width='" + style.width + "' ></canvas>";
  outputImgDiv.innerHTML = html;
}

async function onRunFNSInfer () {
  const styleIdx = document.getElementById("styleSelect").value;
  const style = getStyleList()[styleIdx];

  const onnxModelUrl = style.model_url;
  
  //const backend = document.getElementById("backendSelect").value;
  const backend = 'webgl';

  inferResultsDiv.innerHTML = "<textarea id='inferResultsText' readonly cols=90 rows=10></textarea>";

  // when on desktop OS, cache inference session so it can be re-used for newly selected content image
  // when on mobile OS, Android, do not cache inference session (an issue that inference output is incorrect when cached)
  //      iOS is not yet supported by ONNX.js as of version 0.1.5.
  if (g_onnxSess == null || isMobile()) {
    g_onnxSess = new onnx.InferenceSession({backendHint: backend});
    await g_onnxSess.loadModel(onnxModelUrl);
  }

  //inferResultStr = "loading onnx model..." + newLine;
  inferResultStr = "";
  asyncSetHtml(inferResultsText, inferResultStr).then( ()=>{
    //g_onnxSess.loadModel(onnxModelUrl).then( ()=> {
      srcTensor = canvasToTensor(srcCanvasId);

      inferResultStr += "running fast neural style..." + newLine;
      asyncSetHtml(inferResultsText, inferResultStr).then( ()=>{
        g_onnxSess.run([srcTensor]).then( (pred)=>{
          const output = pred.values().next().value;  // consume output this way so no need to specify output node name.
                                                      //    only for the case of single output node. 
          
          // set output canvas
          tensorToCanvas (output, dstCanvasId);
          
          inferResultStr += "done" + newLine;
          asyncSetHtml(inferResultsText, inferResultStr).then( ()=>{
          });
        });
      });
    //});
  });
}



//-------------------------------------------------------------------------------------------------
// benchmark html
//-------------------------------------------------------------------------------------------------

//
// html functions
//
function htmlBench_GenerateStyleList(list) {
  styleBenchSelect.innerHTML = "";
  for (i=0; i<list.length; i++) {
    if (i==0) {
      styleBenchSelect.innerHTML += "<option value='" + i + "' selected='selected'>" + list[i].style_name + "</option>";
    } else {
      styleBenchSelect.innerHTML += "<option value='" + i + "'>" + list[i].style_name + "</option>";
    }
  }

  htmlBench_GenerateInferStyle(0);
}

function htmlBench_GenerateInferStyle(styleIdx) {
  style = benchmark_getStyleList()[styleIdx];

  // generate HTML
  var html = "";

  html += "<img src='"   + style.thumb_url + "' height=" + style.height + " />" + "&nbsp;";
  html += "<canvas id='" + srcCanvasId     + "' height=" + style.height + " width=" + style.width + " ></canvas>" + "&nbsp;";
  html += "<canvas id='" + dstCanvasId     + "' height=" + style.height + " width=" + style.width + " ></canvas>" + "&nbsp;";

  inferStyleDiv.innerHTML = html;

  // load source image
  //imgUrl = style.content_url;
  imgUrl = g_benchmark_contentImgUrl;

  loadImage (imgUrl, srcCanvasId);

  // fill dest canvas to green
  setCanvasRGB (dstCanvasId, 0x00, 0xFF, 0x00);
}

function htmlBench_onBackendChange() {
  // re-generate style list
  const style_list = benchmark_getStyleList();
  htmlBench_GenerateStyleList(style_list);
}

function htmlBench_onStyleSelectChange() {
  g_onnxSess = null;

  var styleIdx = document.getElementById("styleBenchSelect").value;
  htmlBench_GenerateInferStyle(styleIdx);
}

// do FNS benchmark asynchronously
function htmlBench_onRunFNSBenchmark() {
  const styleIdx = document.getElementById("styleBenchSelect").value;
  const style = benchmark_getStyleList()[styleIdx];

  const onnxModelUrl = style.model_url;
  
  const backend = document.getElementById("backendSelect").value;
  g_onnxSess = new onnx.InferenceSession({backendHint: backend}); // always start a new inference session for benchmark

  // reset benchmark output
  inferTimeList = [];
  
  inferCountDown = totalInferCount;
  
  // disable UI
  styleBenchSelect.disabled = true;
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
  g_onnxSess.loadModel(onnxModelUrl).then( ()=>{
    var loadModelTStr = formatFloat(performance.now() - loadModelT0) + "ms";
    inferResultStr += "load time: " + loadModelTStr + newLine;

    inferResultStr += "warming up tensors... " + newLine;

    asyncSetHtml(inferResultsText, inferResultStr).then( ()=>{

      // warmup tensor
      warmTensor = canvasToTensor(srcCanvasId);
  
      const warmT0 = performance.now();
      g_onnxSess.run([warmTensor]).then( (output)=> {
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

  styleBenchSelect.disabled = false;
  runInferButton.disabled = false;
}

// benchmark
function runFNSCount(){
  // reset output canvas color
  var p = new Promise ((resolve, reject) => {
    setCanvasRGB(dstCanvasId, dstCanvas_r, dstCanvas_g, dstCanvas_b);  // clear the output canvas
    setTimeout (resolve, 10);
  });

  p.then( async ()=>{
    inputTensor = canvasToTensor(srcCanvasId);

    const inferT0 = performance.now();
    pred = await g_onnxSess.run([inputTensor]);
    const inferTime = performance.now() - inferT0;
    const inferTimeStr = formatFloat(inferTime) + "ms";

    const output = pred.values().next().value;
    
    tensorToCanvas (output, dstCanvasId);

    
    inferResultStr += "inference time #" + (inferTimeList.length) + ": " + inferTimeStr + newLine;
    inferResultsText.innerHTML += inferResultStr;
    
    inferCountDown--;
    if (inferCountDown == 0){
      FNSRunCompleteCallback();
    } else {
      // give the output canvas some time (inferDisplayTime) to display
      setTimeout( ()=>{
        runFNSCount();
      }, inferDisplayTime);
    }
    /*
    // run inference
    const inferT0 = performance.now();
    g_onnxSess.run([inputTensor]).then((pred)=>{
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
    */
  });
}