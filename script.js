let tfliteModel = null;

// Load the TensorFlow Lite model
async function loadModel() {
    tfliteModel = await tflite.loadTFLiteModel('mofdel\face_recognition_model_quantized (1).tflite');
    console.log("TFLite model loaded successfully!");
}

// Load the image selected by the user
function loadImage(event) {
    const reader = new FileReader();
    reader.onload = function() {
        const img = new Image();
        img.src = reader.result;
        img.onload = function() {
            const canvas = document.getElementById('imageCanvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            // Display the image
            document.getElementById('inputImage').src = reader.result;
            document.getElementById('inputImage').style.display = 'block';
        };
    };
    reader.readAsDataURL(event.target.files[0]);
}

// Preprocess the image for the TFLite model
function preprocessImage() {
    const canvas = document.getElementById('imageCanvas');
    const imgTensor = tf.browser.fromPixels(canvas)
        .resizeNearestNeighbor([224, 224])  // Resize to the model's input shape
        .toFloat()
        .expandDims(0)  // Add batch dimension
        .div(tf.scalar(255.0));  // Normalize the pixel values
    return imgTensor;
}

// Run the TFLite model and display the result
async function runModel() {
    if (!tfliteModel) {
        alert("Model is not loaded yet!");
        return;
    }

    // Preprocess the image and get the input tensor
    const inputTensor = preprocessImage();

    // Run the model inference
    const outputTensor = tfliteModel.predict(inputTensor);

    // Display the result
    const result = outputTensor.dataSync();  // Convert tensor to array
    document.getElementById('result').innerText = "Prediction: " + result;
}

// Load the model when the page loads
window.onload = function() {
    loadModel();
};
