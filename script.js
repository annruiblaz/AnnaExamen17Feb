// more documentation available at
// https://github.com/tensorflow/tfjs-models/tree/master/speech-commands

// the link to your model provided by Teachable Machine export panel
const URL = "http://127.0.0.1:5500/Modelo/";

//Almacenamos ref al HTMLElement d video 
const video = document.getElementById('videoElement');

async function createModel() {
    const checkpointURL = URL + "model.json"; // model topology
    const metadataURL = URL + "metadata.json"; // model metadata

    const recognizer = speechCommands.create(
        "BROWSER_FFT", // fourier transform type, not useful to change
        undefined, // speech commands vocabulary feature, not useful for your models
        checkpointURL,
        metadataURL);

    // check that model and metadata are loaded via HTTPS requests.
    await recognizer.ensureModelLoaded().then(
        console.log('Modelo Creado')
    );

    return recognizer;
}

createModel();

async function init() {
    const recognizer = await createModel();
    const classLabels = recognizer.wordLabels(); // get class labels
    const labelContainer = document.getElementById("label-container");
    for (let i = 0; i < classLabels.length; i++) {
        labelContainer.appendChild(document.createElement("div"));
    }

    // listen() takes two arguments:
    // 1. A callback function that is invoked anytime a word is recognized.
    // 2. A configuration object with adjustable fields
    recognizer.listen(result => {
        const scores = result.scores; // probability of prediction for each class
        // render the probability scores per class
        for (let i = 0; i < classLabels.length; i++) {
            const classPrediction = classLabels[i] + ": " + result.scores[i].toFixed(2);
            labelContainer.childNodes[i].innerHTML = classPrediction;

            if (result.scores[i].toFixed(2) > 0.7) {
                switch (classPrediction) {
                    case "Preparados":
                        console.log('La palabra detectada es Preparados');
                        break;
                    case "Preparado":
                        console.log('La palabra detectada es Preparado');
                        break;
                    case "Foto":
                        console.log('La palabra detectada es Foto');
                        break;
                    case "Ruido de fondo":
                        console.log('La palabra detectada es Ruido de fondo');
                        break;
                    default:
                        console.log('Default, Ni idea q ha detectado');
                }
            }
        }
    }, {
        includeSpectrogram: true, // in case listen should return result.spectrogram
        probabilityThreshold: 0.75,
        invokeCallbackOnNoiseAndUnknown: true,
        overlapFactor: 0.50 // probably want between 0.5 and 0.75. More info in README
    });

    // Stop the recognition in 5 seconds.
    // setTimeout(() => recognizer.stopListening(), 5000);
}

async function getAccessWebcam() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({
                audio: true,
                video: true
            })
            .then(function (stream) {
                video.srcObject = stream;
            }).catch(function (error) {
                console.error('Se ha producido un error al acceder a la webcam. :(');
            });
        return;
    }
    console.error('Tu navegador no es compatible.');
}

async function getAudioAndWebcam() {
    init();
    getAccessWebcam();
}
