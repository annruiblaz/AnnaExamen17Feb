//Import de MoveNet
const MOVENET_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4';

//Declaramos la variable global para almacenar el modelo
let movenet = undefined;

//Ruta al modelo de Teachable Machine
const URL = "http://127.0.0.1:5500/Modelo/";
//Almacenamos ref al HTMLElement d video 
const video = document.getElementById('videoElement');

//Creamos el modelo d Teachable Machine q guardamos en local
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

//Inicializa el modelo de comandos d voz (una vez creado)
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

            //Nos aseguramos q tiene una certeza decente para hacer la foto
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

//Para acceder a la webcam y audio
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

//inicializa la webcam y el modelo d comandos d voz
async function getAudioAndWebcam() {
    init();
    getAccessWebcam();
}

//carga e inicializa el modelo d MoveNet
async function loadMovenet() {
    //Lo cargamos dsd TensorFlow Hub
    movenet = await tf.loadGraphModel(MOVENET_PATH, {fromTFHub: true});

}
