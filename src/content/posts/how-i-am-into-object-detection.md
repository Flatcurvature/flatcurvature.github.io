---
title: How I Actually Started Into the Object Detection
published: 2025-07-08
description: A hands-on introduction to object detection using YOLO and Darknet, exploring how real-time computer vision works and how it can be applied in practical, everyday scenarios — from robotics to recognizing traffic and food. Also includes ethical reflections on the technology's impact.

tags: [Computer Vision, Machine Learning, AI]
category: AI
draft: false
---


This time I will write the most basic fundamental, maybe younger people like you never encountered Darknet and YOLO Object Detection, so I want to give some tutorial on it since it is a fundamental one. I knew them from humanoid robot research, which needs computer vision and machine learning to detect an object. Since I am a big fan of C++ (to differ me upon what so-called as “modern-time data scientist hype guys”, I know how to code in C++ and actually I am fast). I am a big fan of this model, and I would like to introduce it.

## Darknet

To install darknet you can follow the tutorial on 
[https://pjreddie.com/darknet/install/](PJReddie) website. There is actually [a story on him about why he spent some time decided to not working on AI](https://medium.com/syncedreview/yolo-creator-says-he-stopped-cv-research-due-to-ethical-concerns-b55a291ebb29), which is heartbreaking. However, glad to say he is back since we need more people like him in the research.

Yolo is a real-time object detection system. It uses the dataset to train what contained in an image. Not just image, you could capture the moment from webcam, like a humanoid robot did (Yeah I miss them so much).

![YOLO Performance](https://miro.medium.com/v2/resize:fit:600/format:webp/0*0UQS9CoJsDYaVqie.png)

## How YOLO Works?
YOLO is a **real-time object detection system** that frames detection as a **single regression problem** from image pixels to bounding boxes and class probabilities.

---

## How YOLO Works (YOLOv1–v3)

1. **Divide the Image**:
   - The input image is divided into an **S × S grid** (e.g., 13×13).
   - Each grid cell is responsible for detecting objects whose centers fall into it.

2. **Each Grid Cell Predicts**:
   - **B bounding boxes**, each with:
     - Coordinates: `(x, y, w, h)`
     - **Confidence score**: `Pr(Object) × IOU(pred, truth)`
   - **C class probabilities**
   - Total output per cell:  
     ```
     B × 5 + C
     ```

3. **Single CNN Forward Pass**:
   - A single convolutional neural network processes the image and outputs all predictions in one go.

---

### YOLO vs R-CNN-based Models

| Feature        | YOLO                            | R-CNN / Fast R-CNN / Faster R-CNN          |
|----------------|----------------------------------|---------------------------------------------|
| Input          | Full image once                 | Region proposals (multiple passes)          |
| Speed          | Very fast                       | Slower                                       |
| Accuracy       | Less accurate for small objects | More accurate in some scenarios             |
| Architecture   | Single CNN                      | Region proposals + CNN + classifier         |

---

### YOLOv3: Key Improvements

- **Multi-scale predictions**:
  - Uses feature maps at different layers to detect small, medium, and large objects.

- **Bounding Box Prediction**:
  - Uses anchor boxes, similar to Faster R-CNN.

- **Backbone Network**:
  - Uses Darknet-53 (53 convolutional layers) as the feature extractor.

- **Objectness Score**:
  - Class probabilities are predicted conditionally on the objectness score.

## Into the Practice

I also took the technical step from here:  
https://pjreddie.com/darknet/yolo/

### 1. Clone the GitHub repo

```bash
git clone https://github.com/pjreddie/darknet
```

### 2. Go into the cloned directory, and make it

```bash
cd darknet
make
```

### 3. Download the pre-trained weights

In this example, I will use the YOLOv3 weights, but you could explore more weights on the YOLO page.

```bash
wget https://pjreddie.com/media/files/yolov3.weights
```

### 4. Run the detector on a sample image

You can change `data/dog.jpg` to any other image you'd like to test.

```bash
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

This will produce a detection result using the pre-trained model!

![Jakarta](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*P8IWwsQGQje4AMlpHVj9kw.jpeg)*Figure: Traffic in Jakarta ([source](https://en.wikipedia.org/wiki/Transport_in_Jakarta))*


![Indonesian Food](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*JB-FwebNFt9ByvWSEd_v4A.jpeg)*Figure: Indonesian Food ([source](https://thehoneycombers.com/singapore/indonesian-restaurants-in-singapore-where-to-get-your-rendang-nasi-goreng-tahu-telor-and-other-traditional-indonesian-food/))*

![Cat medieval](https://miro.medium.com/v2/resize:fit:370/format:webp/1*wrG31mRhKMYoHW4IOA-EuQ.jpeg)*Figure: Medieval Painting ([source](https://id.pinterest.com/pin/465489311482997333/))*

## Summary and What’s Next?

In this post, we’ve explored the foundations of real-time object detection using **YOLO (You Only Look Once)**, one of the most impactful and widely used object detection models. Starting from understanding how YOLO works — dividing the image into grids and predicting bounding boxes using a single forward pass — we walked through its evolution up to YOLOv3 and practiced running it using the original Darknet framework.

You also saw how powerful pre-trained models can be in detecting objects in real-world images, such as Jakarta’s traffic, Indonesian food, or even artwork. These examples highlight YOLO’s capability to work across domains, whether in robotics, urban monitoring, or cultural data.

But this is just the beginning.

### What’s Next?

If you’re curious to go further, here are a few directions:

- **Train YOLO on your own dataset**: Try annotating your own images and training a custom object detector.
- **Explore newer YOLO versions**: YOLOv4, YOLOv5, and YOLOv8 have brought in new techniques like data augmentation, transfer learning, and real-time edge deployment.
- **Switch to PyTorch or TensorFlow**: Modern YOLO variants are implemented in more flexible frameworks than Darknet.
- **Dig into applications**: From autonomous systems to assistive devices, object detection is a key part of broader AI systems — and a skill that opens many doors.

And as we’ve discussed in the epilogue, don't forget to reflect on the **social implications** of the tools you build.

## Reference

```
Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
```

## Epilogue

This post is a refined and structured version of [my earlier Medium article](https://medium.com/salmanhiro/how-i-actually-into-the-object-detection-3eee01a44dc1), where I first documented my experience experimenting with YOLO for object detection.

My aim here has been not only to walk through the technical setup — from cloning the Darknet repository to detecting objects with YOLOv3 — but also to place that journey in a wider context. As we build more capable AI systems, especially in computer vision, it's crucial to stay aware of their societal impact.

YOLO, by design is efficient and real-time to enable applications in self-driving cars, medical imaging, drone footage analysis, and unfortunately, also in surveillance and military tech. This ethical problem was one of the reasons **Joseph Redmon**, the original creator of YOLO, made the decision to leave the field. He said:

> “I stopped doing CV research because I saw the impact my work was having. I loved the work but the military applications and privacy concerns eventually became impossible to ignore.”

This moment invites us to reflect. We often celebrate breakthroughs in AI without asking who benefits, who gets left out, and who could be harmed. Redmon’s message is a reminder that even well-intentioned open research can have unintended consequences when deployed at scale.

So while this guide is practical in nature, I hope it also encourages thoughtful curiosity -- not just about how things work, but about why we build them, and what kind of better world we want to build with them.

---


