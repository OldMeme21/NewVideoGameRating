
async function runExample() {

    var x = new Float32Array( 1, 24 )

    var x = [];

    x[0] = document.getElementById('alcohol_reference').value;
    x[1] = document.getElementById('animated_blood').value;
    x[2] = document.getElementById('blood').value;
    x[3] = document.getElementById('blood_and_gore').value;
    x[4] = document.getElementById('cartoon_violence').value;
    x[5] = document.getElementById('crude_humor').value;
    x[6] = document.getElementById('drug_reference').value;
    x[7] = document.getElementById('fantasy_violence').value;
    x[8] = document.getElementById('intense_violence').value;
    x[9] = document.getElementById('language').value;
    x[10] = document.getElementById('lyrics').value;
    x[11] = document.getElementById('mature_humor').value;
    x[12] = document.getElementById('mild_blood').value;
    x[13] = document.getElementById('mild_cartoon_violence').value;
    x[14] = document.getElementById('mild_fantasy_violence').value;
    x[15] = document.getElementById('mild_language').value;
    x[16] = document.getElementById('mild_lyrics').value;
    x[17] = document.getElementById('mild_suggestive_themes').value;
    x[18] = document.getElementById('mild_violence').value;
    x[19] = document.getElementById('no_descriptors').value;
    x[20] = document.getElementById('nudity').value;
    x[21] = document.getElementById('partial_nudity').value;
    x[22] = document.getElementById('sexual_content').value;
    x[23] = document.getElementById('sexual_themes').value;

    let tensorX = new onnx.Tensor(x, 'float32', [1, 24]);

    let session = new onnx.InferenceSession();

    await session.loadModel("./esrb_classifier.onnx");
    let outputMap = await session.run([tensorX]);
    let outputData = outputMap.get('output');

   let predictions = document.getElementById('predictions');

   let ratings = ["E", "ET", "T", "M"];
   let ratingIndex = Math.round(outputData.data[0]);

predictions.innerHTML = ` <hr> Game Rating Prediction: <br/>
   <table>
     <tr>
       <td>  Predicted Rating  </td>
       <td id="td0">  ${ratings[ratingIndex]}  </td>
     </tr>
  </table>`;
    


}


