# CaptchaSAM

### 1. Character-level semi-automatic annotation using SAM
Annotation tool: 
- https://github.com/yatengLG/ISAT_with_segment_anything

Skip the annotation step by downloading Captcha images and corresponding annotations at [Google Drive](https://drive.google.com/file/d/1jLS59jmtsaRQeCtX5TF2kAIUYsBADX8Z/view?usp=sharing)

### 2. Convert annotations into Yolo format 
`python convert.py`

### 3. Train Yolo Models
```
cd YoloSeg
python main.py
```

The Captcha images, annotations in Yolo format, and trained models can be downloaded at [Google Drive](https://drive.google.com/file/d/1EGIEqVzwAZo02Mis7IKt4Lddy3xO29h9/view?usp=sharing)

### 4. Captcha Recognition
```
cd YoloSeg
python predict.py
```




