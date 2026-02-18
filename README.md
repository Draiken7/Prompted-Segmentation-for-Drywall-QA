# Prompted Segmentation for Drywall QA
## Goal
Train (or at least fine-tune) a text-conditioned segmentation model so that, given an image and a natural-language prompt, it produces a binary mask for:
● “segment crack” (Dataset 2: Cracks)
● “segment taping area” (Dataset 1: Drywall-Join-Detect)

## Methodology
### Segmentation
For prompted segmentation on images of wall cracks, I tried a simple approach of training a pretrained model to associate some prompts as text embeddings to labelled segmentation data. Essentially, the problem can be broken into two parts where one is associating a prompt or embedding to the image and segmentation mask and secondly learning segmentation on images. The model to use was informed by limited time and compute power to train it, so I preferred the Hugging face CLIPSeg model.

#### Dataset and DataPreparation
Dataset for this is hosted at roboflow: cracks [https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36]
Since the dataset is hosted at roboflow, there was no extra data preparation needed except for transformations and sanitization required by the model to be trained. The final dataset used also had augmented data with orientation-based changes made to augment the dataset, like rotation, shear, flip and so on.

#### Model (CLIPSeg)
CLIPSeg was used because of its simplicity and ease of use in limited resource environment. The model commonly uses a ViT-B/16 CLIP backbone with nearly 1.1 million trainable parameters. The model is also achieves a general mIoU of 0.23 when doing zero shot prediction and few shot fine tuning can improve this. While training on native machine with a Nvidia 3060 with 6gb vram, the model was trained for 7 epochs in 1.5 hours with inference run on nearly 100 test images that can be done within a minute with an average inference time of 0.6 seconds per image. The trained model size in ~600 MB. 

#### Performance
The model was able to achieve the below given performance metrics. Some examples of the actual segmentation data is also attached below

<img width="275" height="62" alt="image" src="https://github.com/user-attachments/assets/819ebe0c-9dd5-4b58-b42b-2447dbabedf0" />
<img width="174" height="58" alt="image" src="https://github.com/user-attachments/assets/1fff2dee-8067-48e6-8f43-393e376462bd" />

<img width="1982" height="1014" alt="output_9_test" src="https://github.com/user-attachments/assets/0ff52f1a-e7d2-4039-8329-3f785f22da70" />
<img width="1982" height="1014" alt="output_81_test" src="https://github.com/user-attachments/assets/1ab80e73-7fe1-4c5b-9af3-80c16d743c34" />
<img width="1982" height="1014" alt="output_64_test" src="https://github.com/user-attachments/assets/05da74fb-85a0-47f2-a4b8-5fa732184686" />
<img width="1982" height="1014" alt="output_53_test" src="https://github.com/user-attachments/assets/d5c08ab5-cb11-4031-96b2-e2753427c88e" />
<img width="1982" height="1014" alt="output_11_test" src="https://github.com/user-attachments/assets/320bd3f1-d2aa-4267-a0f3-7c083810482d" />

The model seems to struggle with fine cracks. This can be handled by:
1. Using Sliding window, although this may affect the inference speed.
2. Augmenting dataset with more images showing clearer fine cracks.
Obfuscated cracks may also be failure points, for which augmenting data with by adjusting contrast and exposure might help the model. All in all, the model is a good base point to start at due to its simplicity yet effective performance.


### Object Detection
For prompted object detection on wall segments and joints, the approach remained the same – learning to associate prompt to given image and bounding box while also learning to detect required object in the image. So initially, I expected to use OWLv2 model since it is a open vocabulary object detection model, but due to time and compute restraints, decided to change my approach. Since the prompts I was working with were limited for this experiment, I decided to use YOLOWorld instead with the prompt being fed as custom classes. This is not a generally good approach, but due to constraints, this seemed like a workable, if restrictive, option.

#### Dataset and DataPreparation
Dataset for this is hosted at roboflow: joint-detect [https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect]
Since the dataset is hosted at roboflow, there was no extra data preparation required., The final dataset used also had augmented data with orientation-based changes made to augment the dataset, like rotation, shear, flip and so on with a tiling preprocessing happening by mistake which may have resulted in the poor performance of the model. The dataset was significantly smaller with missing labels for images after augmentation and preprocessing. Handling this is essential before retraining the model to improve performance.

#### Model (YOLOWorld)
YOLOWorld is a larger model with nearly 11 million parameters which makes it significantly time consuming to train. Generally, YOLO achieves a mAP score of 44% on the standard COCO dataset. The model was trained on native machine using Nvidia 3060 with 6GB vram and took upwards of 2.5 hours to train for nearly 12 epochs (The training was done in batches of 5 and 7 epochs). The trained model has the size nearly 25MB. The model is slow to train, and only uses a limited set of prompts as given below as class names to associate with the bounding boxes:
["segment taping area", "segment joint/tape", "segment drywall seam", "segment drywall tape joint", "segment wallboard joint area"]
This method is not great but was chosen due to time and compute restraints in place of OWLv2 model which is extremely memory heavy. Speed: 0.2ms preprocess, 4.4ms inference, 0.0ms loss, 1.2ms postprocess per image.

#### Performance
The model achieves the following performance metric on train and validation sets:
 <img width="1044" height="54" alt="image" src="https://github.com/user-attachments/assets/349130c5-ab3c-4638-801b-5938e23f25eb" />

The model is not very efficient but achieves the following metrics on test dataset:
  <img width="286" height="61" alt="image" src="https://github.com/user-attachments/assets/9cfbf8ea-9765-4458-abbb-b886ee70f4f9" />

The following are the Ground truth and predicted bounding boxes for random test images:
<img width="1212" height="528" alt="YOLO_3" src="https://github.com/user-attachments/assets/bd948174-aff6-4dc0-ad7b-4e547d5f1b03" />
<img width="1212" height="562" alt="YOLO_2" src="https://github.com/user-attachments/assets/580f1585-721e-4201-8dba-851dcae8d856" />
<img width="1210" height="558" alt="YOLO_1" src="https://github.com/user-attachments/assets/4950f11c-86ce-40a1-a3a5-593a1e76e7bb" />
<img width="1206" height="529" alt="YOLO_4" src="https://github.com/user-attachments/assets/6b92a44f-e4df-44b2-ad34-8c6267c23da9" />

The primary cause of failure comes from first the tiling that divides each image into smaller chunks (each image is divided into 4x4 grid and passed) and the orientation change which add black background to the image. The second noticeable failure points are fine joints, especially on walls with many marks obfuscating what is the joint and what are just scratches and marks. 
In this case, firstly, a different model needs to be found that can handle prompt association with object detection rather than mapping to new classes. Secondly, data augmentation needs to avoid orientation changes that add black borders that may be misconstrued for a different section of the wall. The failure points might remain but based on the performance, methods like data augmentation using changes in contrast and exposure might help.




