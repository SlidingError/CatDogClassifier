const fileInput = document.getElementById("fileInput");
const previewBox = document.getElementById("previewBox");
const previewImage = document.getElementById("previewImage");

const resultBox = document.getElementById("resultBox");
const labelText = document.getElementById("labelText");
const confidenceText = document.getElementById("confidenceText");

const loadingText = document.getElementById("loadingText");
const predictBtn = document.getElementById("predictBtn");


// Preview image after selection
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];

    if (!file) return;

    const imageURL = URL.createObjectURL(file);

    previewImage.src = imageURL;
    previewBox.classList.remove("hidden");
    resultBox.classList.add("hidden");
});


// Predict button click
predictBtn.addEventListener("click", predictImage);


// Send image to backend
async function predictImage() {
    const file = fileInput.files[0];

    if (!file) {
        alert("Please choose an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    loadingText.classList.remove("hidden");
    resultBox.classList.add("hidden");
    predictBtn.disabled = true;

    try {
        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        labelText.innerText =
            "Label: " + data.prediction;

        confidenceText.innerText =
            "Confidence: " +
            (data.confidence * 100).toFixed(2) + "%";

        resultBox.classList.remove("hidden");

    } catch (error) {
        alert(error.message);
    }

    loadingText.classList.add("hidden");
    predictBtn.disabled = false;
}