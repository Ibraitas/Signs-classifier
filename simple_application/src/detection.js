import React from "react";
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';
import "./styles.css";
import classesDir from './signsObjects';
tf.setBackend('webgl');

const threshold = 0.75;

async function load_model() {
    const model = await loadGraphModel("https://raw.githubusercontent.com/Ibraitas/Signs-classifier/master/web_model/model.json");
    return model;
  }

class Detection extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            detections: []
        };
    }
  videoRef = React.createRef();
  canvasRef = React.createRef();


  componentDidMount() { //TODO: Delete "!"
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "user"
          }
        })
        .then(stream => {
          window.stream = stream;
          this.videoRef.current.srcObject = stream;
          return new Promise((resolve, reject) => {
            this.videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          });
        });

      const modelPromise = load_model();

      Promise.all([modelPromise, webCamPromise])
        .then(values => {
          this.detectFrame(this.videoRef.current, values[0]);
        })
        .catch(error => {
          console.error(error);
        });
    }
  }

    detectFrame = (video, model) => {
        tf.engine().startScope();
        model.executeAsync(this.process_input(video)).then(predictions => {
        this.renderPredictions(predictions, video);
        requestAnimationFrame(() => {
          this.detectFrame(video, model);
        });
        tf.engine().endScope();
      });
  };

  process_input(video_frame){
    const tfimg = tf.browser.fromPixels(video_frame).toInt();
    const expandedimg = tfimg.transpose([0,1,2]).expandDims();
    return expandedimg;
  };

  buildDetectedObjects(scores, threshold, boxes, classes, classesDir) {
    const detectionObjects = []
    var video_frame = document.getElementById('frame');

    scores[0].forEach((score, i) => {
      if (score > threshold) {
        const bbox = [];
        const minY = boxes[0][i][0] * video_frame.offsetHeight;
        const minX = boxes[0][i][1] * video_frame.offsetWidth;
        const maxY = boxes[0][i][2] * video_frame.offsetHeight;
        const maxX = boxes[0][i][3] * video_frame.offsetWidth;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        detectionObjects.push({
          class: classes[i],
          label: classesDir[classes[i]].name, //TODO: ADD LINKS
          score: score.toFixed(4),
          bbox: bbox
        })
      }
    })
    return detectionObjects
  }

  renderPredictions = predictions => {
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    //Getting predictions
    const boxes = predictions[1].arraySync();
    const scores = predictions[4].arraySync();
    const classes = predictions[2].dataSync();
    const detections = this.buildDetectedObjects(scores, threshold,
                                    boxes, classes, classesDir);
    this.setState({ detections });
    detections.forEach(item => {
    console.log(item);
      const x = item['bbox'][0];
      const y = item['bbox'][1];
      const width = item['bbox'][2];
      const height = item['bbox'][3];

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%").width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(item["label"] + " " + (100*item["score"]).toFixed(2) + "%", x, y);
    });
  };

  render() {
    const { detections } = this.state;
    return (
      <div className="main-container">
        <h1 className="title">Real-Time Object Detection: Signs Classifier</h1>
        <div className="button-container">
            { detections.map(item => (<button className="sign-link">
                <a href={ item.label }>{ item.label }</a>
            </button>)) }
        </div>
        <div className="video-detect">
            <video
            className="size"
            autoPlay
            playsInline
            muted
            ref={this.videoRef}
            width="600"
            height="500"
            id="frame"
            />
            <canvas
            className="size"
            ref={this.canvasRef}
            width="600"
            height="500"
        />
        </div>
      </div>
    );
  }
}

export default Detection;