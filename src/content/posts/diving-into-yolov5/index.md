---
title: Diving into Latest YOLOv5
published: 2020-08-26
image: "./cover.jpeg"
description: A next version of YOLO v5, the development of one of the most popular object detection models in the computer vision community
tags: [Computer Vision, Machine Learning, AI]
category: AI
draft: false
---

> Reading time: 15 minutes

> Cover image source: [Source](https://www.pixiv.net/en/artworks/130934677)


Following my previous post about [YOLOv3 object detection](https://flatcurvature.github.io/posts/how-im-into-object-detectionmd/), this time I jumped into YOLOv5 (yes I tried the v4, but not yet hyped so I tried the v5) for object detection.

YOLOv5 has pretrained model which included 80 classes as the previous model has. I found that the way YOLOv5 presented in PyTorch is very similar to the Darknet version, although controversially this one seems like a PyTorch implementation from previous version but lighter (spill some tea, the original YOLO ended in v3). Here is the [Ultralytics YOLOv5 repositories](https://github.com/ultralytics/yolov5?source=post_page-----6a3fc33d4931---------------------------------------).

To begin with YOLOv5, you’ll need to clone the repository locally. Alternatively, there's a Google Colab version available (refer to the README for the link).

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
```

The repository provides several tutorials, but let’s jump straight into testing the object detection feature.

First Prediction
Before running detection, grab any image you like. For instance, here’s using one from Pinterest. To detect objects, run

```bash
python detect.py <IMG_FILENAME>
```



We should expect the model to detect a person and a cat in this image. By default, YOLOv5 will use the yolov5s model — the smallest and fastest version.

![result1](https://miro.medium.com/v2/resize:fit:926/format:webp/1*Pf94N1hyDmunrOw_NEhhTw.jpeg)

Image: Pinterest [Source](pinterest.com/pin/765049055437540826/?nic_v2=1a7nczwII)

Another try, 

![result2](https://miro.medium.com/v2/resize:fit:960/format:webp/1*rTg-qEPg7GtR6y67Axk6iA.jpeg)

Image: Pinterest [Source](pinterest.com/pin/765049055437540826/?nic_v2=1a7nczwII)

![result3](https://miro.medium.com/v2/resize:fit:1128/format:webp/1*3rFhbenYzhZ-pKiYtw9SQQ.jpeg)

Image: Pinterest [Source](https://id.pinterest.com/pin/62276407332452555/?nic_v2=1a7nczwII)

## Comparison to YOLOv3

While YOLOv3 has been widely used and appreciated for its accuracy and speed, YOLOv5 brings several improvements that are hard to ignore:

| Feature            | YOLOv3                           | YOLOv5                            |
|--------------------|----------------------------------|-----------------------------------|
| Framework          | Darknet (C/CUDA)                 | PyTorch (Python)                  |
| Deployment Ease    | Requires compilation             | Easy to set up and run in Python |
| Model Variants     | Tiny and Full                    | s, m, l, x (scalable architecture) |
| Inference Speed    | Slower, especially on CPU        | Faster and lightweight           |
| Training Workflow  | Manual data pipeline             | Built-in training, validation, logging |
| ONNX Export        | Not native                       | Supported out of the box         |

From both a development and usability perspective, YOLOv5 is clearly built with accessibility and speed in mind, especially for those already familiar with the PyTorch ecosystem.

## Conclusion

YOLOv5 shows promise for its usability, which combined with PyTorch makes it an easier option for modern computer vision tasks.

While it may be controversial ([there is an interesting writing](https://medium.com/augmented-startups/yolov5-controversy-is-yolov5-real-20e048bebb08)) due to its unofficial status in the YOLO family, since I saw it like the original project was abandoned because of the owner's concern, then a company taking it over for their business product, YOLOv5 has some practical performance. For anyone getting into real-time object detection or prototyping vision applications, it's worth exploring.

---