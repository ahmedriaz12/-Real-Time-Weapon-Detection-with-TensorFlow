# Real-Time Weapon Detection with TensorFlow Object Detection API

![Image description](http://github.com/ahmedriaz12/-Real-Time-Weapon-Detection-with-TensorFlow/blob/master/proj-demo/de1.gif)

## Problem Statement
Over the past decade, the frequency of mass shootings in schools has increased dramatically. Gun violence on school grounds has left many children injured or killed, reflecting a wider issue of firearm-related violence in the United States. As long-term solutions are debated, there is a growing need for immediate alternatives that can help reduce or prevent such incidents.

On average, police response time is around 18 minutes. During an active shooting, calling 911 may not be possible right away—people often need to find safety first, which delays the process. This means the 18 minutes only accounts for response time after the call, not the time it takes to make it.

![Image description](http://github.com/ahmedriaz12/-Real-Time-Weapon-Detection-with-TensorFlow/blob/master/proj-demo/respons_time.png)

## Project Overview
This project explores the use of real-time weapon detection integrated with CCTV cameras to drastically reduce police response time. The object detection model is designed to identify weapons in images, videos, or live video streams.

Once a weapon is detected on a live feed, authorities can be alerted immediately. They can monitor the feed, confirm the threat, and dispatch units without relying on a phone call. This can cut down response times significantly and potentially prevent casualties.

## Detection Methodology
Training an object detection model from scratch is both time- and resource-intensive. Instead, transfer learning provides a more practical approach. TensorFlow’s Object Detection API allows customization of pre-trained models, originally trained on general datasets like COCO, to detect domain-specific objects.

Some of the available pre-trained models include:

| Model name  | Speed (ms) | COCO mAP |
| ------------ | :--------------: | :--------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | 30 | 21 |
| [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) | 31 | 22 |
| [ssdlite_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz) | 27 | 22 |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) | 42 | 24 |
| [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) | 58 | 28 |

For this project, I chose **`ssd_mobilenet_v2_coco`** because:
1. It provides low inference time (31 ms), suitable for real-time webcam input.
2. It offers a reasonable accuracy score (mAP).

## Dataset
The dataset was sourced from the [University of Granada research group](https://sci2s.ugr.es/weapons-detection). It consists of ~3,000 images of firearms (mainly handguns) in `.jpg` format, each with an accompanying annotation file in `.xml` format containing class labels and bounding box coordinates.

Example XML structure:
```xml
<annotation>
  <filename>armas (3)</filename>
  <size>
    <width>1300</width>
    <height>866</height>
    <depth>3</depth>
  </size>
  <object>
    <name>pistol</name>
    <bndbox>
      <xmin>471</xmin>
      <ymin>207</ymin>
      <xmax>613</xmax>
      <ymax>359</ymax>
    </bndbox>
  </object>
</annotation>
```

## Training Process
The model was trained on Google Colab using GPU acceleration. The training lasted around 14 hours and involved fine-tuning hyperparameters, such as:
- **Image augmentation** (rotation, flipping, resizing, color adjustments)
- **Learning rate modifications**
- **Regularization techniques** (e.g., dropout)

TensorBoard was used to monitor performance metrics like loss and Mean Average Precision (mAP) in real time.

## Model Evaluation
The evaluation primarily focused on **mAP** scores and **Intersection over Union (IoU)** thresholds:

![Image description](http://github.com/ahmedriaz12/-Real-Time-Weapon-Detection-with-TensorFlow/blob/master/proj-demo/IOU.png)

A detection is considered **True Positive** if its IoU exceeds the chosen threshold, otherwise it’s classified as **False Positive**.

![Image description](http://github.com/ahmedriaz12/-Real-Time-Weapon-Detection-with-TensorFlow/blob/master/proj-demo/mAP%40.5IOU.png)

## Conclusion
This project demonstrates how AI-powered surveillance can help reduce police response time during firearm-related incidents. While further improvements are needed to minimize false positives, the method shows strong potential as a preventive tool.

Beyond schools, such technology could be deployed in other public and private spaces—parks, banks, gas stations, or shopping centers—providing an additional layer of security.

