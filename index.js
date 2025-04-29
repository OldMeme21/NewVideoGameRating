
async function runExample() {

    let x = [];

    // Collect and parse input values
    x.push(parseFloat(document.getElementById('alcohol_reference').value));
    x.push(parseFloat(document.getElementById('animated_blood').value));
    x.push(parseFloat(document.getElementById('blood').value));
    x.push(parseFloat(document.getElementById('blood_and_gore').value));
    x.push(parseFloat(document.getElementById('cartoon_violence').value));
    x.push(parseFloat(document.getElementById('crude_humor').value));
    x.push(parseFloat(document.getElementById('drug_reference').value));
    x.push(parseFloat(document.getElementById('fantasy_violence').value));
    x.push(parseFloat(document.getElementById('intense_violence').value));
    x.push(parseFloat(document.getElementById('language').value));
    x.push(parseFloat(document.getElementById('lyrics').value));
    x.push(parseFloat(document.getElementById('mature_humor').value));
    x.push(parseFloat(document.getElementById('mild_blood').value));
    x.push(parseFloat(document.getElementById('mild_cartoon_violence').value));
    x.push(parseFloat(document.getElementById('mild_fantasy_violence').value));
    x.push(parseFloat(document.getElementById('mild_language').value));
    x.push(parseFloat(document.getElementById('mild_lyrics').value));
    x.push(parseFloat(document.getElementById('mild_suggestive_themes').value));
    x.push(parseFloat(document.getElementById('mild_violence').value));
    x.push(parseFloat(document.getElementById('no_descriptors').value));
    x.push(parseFloat(document.getElementById('nudity').value));
    x.push(parseFloat(document.getElementById('partial_nudity').value));
    x.push(parseFloat(document.getElementById('sexual_content').value));
    x.push(parseFloat(document.getElementById('sexual_themes').value));

    let tensorX = new onnx.Tensor(new Float32Array(x), 'float32', [1, 24]);

    let session = new onnx.InferenceSession();
    await session.loadModel("./esrb_classifier.onnx");

    // **IMPORTANT**: the input name must match your ONNX model input name.
    // For now I assume it's "input1" — you can check the real name using Netron or ONNX inspection tools.
    let outputMap = await session.run({ input1: tensorX });
    let outputData = outputMap.values().next().value;

    let predictions = document.getElementById('predictions');

    let ratings = ["E", "ET", "T", "M"];

    // **Assume** the model outputs probabilities/scores
    let maxScore = -Infinity;
    let ratingIndex = -1;
    for (let i = 0; i < outputData.data.length; i++) {
        if (outputData.data[i] > maxScore) {
            maxScore = outputData.data[i];
            ratingIndex = i;
        }
    }

    predictions.innerHTML = `
        <hr> Game Rating Prediction: <br/>
        <table>
            <tr>
                <td> Predicted Rating </td>
                <td id="td0"> ${ratings[ratingIndex]} </td>
            </tr>
        </table>`;
}
