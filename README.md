# BurnoutObjectDetection
___
This model was trained with state-of-the-art objection detection algorithm which is SSD with EfficientNet-b0 + BiFPN feature extractor a.k.a EfficientDet (you can read the paper here https://arxiv.org/abs/1708.02002). To be precise, we use EfficientDet D0 512x512 as the pretrained model. We only used 81 data of pcb image burnt to train the model and split those data to 65 images to train set, 8 images to validation set, and 8 images to test set.

![alt text](screenshot/ss1.png)
___
### Clone Project 
```bash
git clone --recursive https://github.com/irvankadhafi/BurnoutObjectDetection.git
```


### Set Environment Variable (Windows : Administrator CMD)
```bash
set PYTHONPATH=<absolute-project-path>\tfod-api;<absolute-project-path>\tfod-api\research;<absolute-project-path>\tfod-api\research\slim
```

### Set Environment Variable (Linux)
```bash
export PYTHONPATH=<absolute-project-path>/tfod-api:<absolute-project-path>/tfod-api/research:<absolute-project-path>/tfod-api/research/slim
```

### How to make virtual environment (required python3 ):
___
#### Running this command in this project folder
```bash
python -m venv ./venv
```
it will created folder named `venv`
#### Activate created environment
_Linux_
```bash
source venv/bin/activate
```
_Windows_ (Using CMD in project folder)
```bash
venv\Scripts\activate.bat
```
___
### Video yang digunakan
[test.mp4](https://drive.google.com/file/d/1-OycRKplMPSQ_kmSsQrU7viWgD79QnEM/view?usp=sharing)
