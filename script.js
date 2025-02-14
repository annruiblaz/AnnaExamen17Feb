//Import de MoveNet
const MODEL_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4';

//Declaramos la variable global para almacenar el modelo
let movenet = undefined;

//Ruta al modelo de Teachable Machine
const URL = "http://127.0.0.1:5500/Modelo/";
//Almacenamos ref al HTMLElement d video y el canvas
const VIDEO_ELEMENT = document.getElementById('videoElement');
const VIDEO_OVERLAY = document.getElementById('videoOverlay');
const ORIGINAL_CANVAS = document.getElementById('canvasPicture');
const CROPPED_CANVAS = document.getElementById('croppedCanvas');



//Carga e inicializa el modelo d MoveNet
async function loadMovenet() {
    //Lo cargamos dsd TensorFlow Hub x eso el true
    movenet = await tf.loadGraphModel(MODEL_PATH, {fromTFHub: true});
}

loadMovenet();

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
        document.getElementById('allowAccess').classList.add('visible')
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
                switch (classLabels[i]) {
                    case "Preparados":
                        readText('Preparados');
                        break;
                    case "Preparado":
                        readText('Preparados');
                        break;
                    case "Foto":
                        readText('Foto');
                        break;
                    case "Ruido de fondo":
                        break;
                    default:
                        break;
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
            .then( function (stream) {
                VIDEO_ELEMENT.srcObject = stream;
            }).catch( function (error) {
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

async function makeImgTensor() {
    //Creamos un tensor con la img q ya está almacenada en el canvas
    let imgTensor = tf.browser.fromPixels(ORIGINAL_CANVAS);
    console.log('Forma del tensor de la imagen: ', imgTensor.shape); // [480, 640, 3]

    //Recortamos la img para q cuadre con el formato esperado x movenet [1, 192, 192, 3]
    //Establecemos el punto d incio para el recorte d la img [y, x, canal] (El canal está a 0 xq queremos q incluya los 3 canales)
    let cropStartPoint = [15, 170, 0];
    //Asignamos el tamaño del recorte [alto, ancho, canales]
    let cropSize = [345, 345, 3];

    //Slice recorta la img dsd el punto de incicio utilizando el tamaño definido
    let croppedTensor = tf.slice(imgTensor, cropStartPoint, cropSize);

    //Mostramos la img recortada en el canvas
    await tf.browser.toPixels(croppedTensor, CROPPED_CANVAS);

    //Como la img continua siendo muy grande, le aplicamos un resizeBilinear
    let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt();
    console.log('Tensor ajustado con su resize:', croppedTensor.shape);

    if(movenet != undefined) {
    //Expandimos las dimensiones para añadir la q falta al comienzo
    let tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
    console.log('Predicciones tensorOutput:', tensorOutput);
    //Convertimos la salida del modelo en un array para visualizar los results
    let arrayOutput = await tensorOutput.array();
    console.log('Predicciones Output array: ', arrayOutput);

    //Verificamos si el output d predicciones tiene keypoints
    if( arrayOutput.length > 0 && arrayOutput[0][0]) {
        let keypoints = arrayOutput[0][0]; //Almacenamos los keypoints
        drawKeypoints(keypoints, CROPPED_CANVAS, cropSize[0] / 192);
            //Mostramos la img recortada en el canvas
        //await tf.browser.toPixels(resizedTensor, CROPPED_CANVAS);

    } else {
        console.error('No se han encontrado keypoints.');
    }
    }

    //Limpiamos los tensores
    imgTensor.dispose();
    croppedTensor.dispose();
    resizedTensor.dispose();
    //tensorOutput.dispose();
}


//Dibujamos los puntos clave en el canvas
function drawKeypoints(keypoints, canvas, scale) {
    const ctx = canvas.getContext("2d");
    // ** ToDo: Lógica q cambie el color del punto en base a la posición d las mannos con la cabezaaaaaaaaaaaa
    ctx.fillStyle = "#4F9D69";
    ctx.lineWidth = 2;

    if(keypoints[9][0] <= keypoints[2][0] || keypoints[10][0] <= keypoints[1][0]) {
        console.log('Manos encima cabeza');
        ctx.fillStyle = "#D84654";
    }

    keypoints.forEach( point => {
        let x = point[1] * 192 * scale;
        let y = point[0] * 192 * scale;

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
    });

    console.log('Acabo d dibujar keypoints');
}

async function takePhoto() {
    const ctxCanvas = ORIGINAL_CANVAS.getContext("2d");
    ORIGINAL_CANVAS.width = VIDEO_ELEMENT.videoWidth;
    ORIGINAL_CANVAS.height = VIDEO_ELEMENT.videoHeight;
    ctxCanvas.drawImage(VIDEO_ELEMENT, 0, 0, ORIGINAL_CANVAS.width, ORIGINAL_CANVAS.height)

    VIDEO_OVERLAY.classList.remove('takePhoto');

    makeImgTensor();
}

async function readText(type) {
    VIDEO_OVERLAY.classList.add('takePhoto');

let message = '';
    switch(type) {
        case "Foto":
            message = 'Patata';
            break;
        case "Preparados":
            message = 'listos ya ¡Patata!';
            break;
        default:
            message = 'Error perooo di patata';
            break;
    }

    //Creamos el obj d síntesis d voz
    const readText = new SpeechSynthesisUtterance(message);

    //configuramos las opciones
    readText.lang = 'es-ES';
    readText.rate = .85;
    readText.pitch = 1;
    readText.volume = 1;

    speechSynthesis.speak(readText);

    readText.onend = () => {
        if(type === 'Preparados') {
            setTimeout(takePhoto, 5000);
        } else {
            takePhoto();
        }
    }
}

