var slider;
let origImages = [
  { "src": "./images/VisualResults/ISO_Chart_1__Decoded_DistgSSR.jpg", "label": "DistgSSR Results", },
  { "src": "./images/VisualResults/ISO_Chart_1__Decoded_EPIT.jpg", "label": "EPIT Results", }
];
let origOptions = {
  "makeResponsive": true,
  "showLabels": true,
  "mode": "horizontal",
  "showCredits": true,
  "animate": true,
  "startingPosition": "50"
};

const juxtaposeSelector = "#juxtapose-embed";
const transientSelector = "#juxtapose-hidden";


function tab_gallery_click(name) {
  // Get the expanded image
  let inputImage = {
    label: "DistgSSR Results",
  };
  let outputImage = {
    label: "EPIT Results",
  };

  inputImage.src = "./images/VisualResults/".concat(name, "_DistgSSR", ".jpg")
  outputImage.src = "./images/VisualResults/".concat(name, "_EPIT", ".jpg")

  let images = [inputImage, outputImage];
  let options = slider.options;
  options.callback = function (obj) {
    var newNode = document.getElementById(obj.selector.substring(1));
    var oldNode = document.getElementById(juxtaposeSelector.substring(1));
    console.log(obj.selector.substring(1));
    console.log(newNode.children[0]);
    oldNode.replaceChild(newNode.children[0], oldNode.children[0]);
    //newNode.removeChild(newNode.children[0]);

  };

  slider = new juxtapose.JXSlider(transientSelector, images, options);
};

(function () {
  slider = new juxtapose.JXSlider(
    juxtaposeSelector, origImages, origOptions);
  //document.getElementById("left-button").onclick = replaceLeft;
  //document.getElementById("right-button").onclick = replaceRight;
})();
// Get the image text
var imgText = document.getElementById("imgtext");
// Use the same src in the expanded image as the image being clicked on from the grid
// expandImg.src = imgs.src;
// Use the value of the alt attribute of the clickable image as text inside the expanded image
imgText.innerHTML = name;
// Show the container element (hidden with CSS)
// expandImg.parentElement.style.display = "block";

$(".flip-card").click(function () {
  console.log("fading in")
  div_back = $(this).children().children()[1]
  div_front = $(this).children().children()[0]
  // console.log($(this).children("div.flip-card-back"))
  console.log(div_back)
  $(div_front).addClass("out");
  $(div_front).removeClass("in");

  $(div_back).addClass("in");
  $(div_back).removeClass("out");

});

$(".flip-card").mouseleave(function () {
  console.log("fading in")
  div_back = $(this).children().children()[1]
  div_front = $(this).children().children()[0]
  // console.log($(this).children("div.flip-card-back"))
  console.log(div_back)
  $(div_front).addClass("in");
  $(div_front).removeClass("out");

  $(div_back).addClass("out");
  $(div_back).removeClass("in");

});