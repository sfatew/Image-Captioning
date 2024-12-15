# Image-Captioning

Generate a short caption for an image

## Enviroment setup
* To clone repository
```
git clone <https://github.com/sfatew/Image-Captioning.git>
cd <Image-Captioning>
```
* To install requirements libraries:
```
pip install -q -r requirements.txt
```
## To run the model:
### Model Checkpoints
The model checkpoints are stored in the drive <https://drive.google.com/drive/folders/14YwsskmkFAd4_RlXLIQ-CBOtrjQTsbMl>

### To run the model

put the models in the drive into the folder `/model`

You can run the each of the following commands to test each models on the image at your working directory

**For cnn&lstm model use :**

firstly, in the **IMAGE_CAPTIONING** folder: Create 2 folder **Data_set** and **working**

secondly, create folder **Model**  inside **working** folder

thirdly, go to drive <https://drive.google.com/drive/folders/14YwsskmkFAd4_RlXLIQ-CBOtrjQTsbMl> to:

- dowload *CoCo_transform_train2017.json* file, PUT INTO **Data_set**
- dowload *VGG&LSTM_CoCo_model.keras* file, PUT INTO **working/Model**
```
python3 transformer_model_infer.py --image_path <path_to_image/image.jepg>
python3 transformer_model_pre_infer.py --image_path <path_to_image/image.jepg>
python3 cnn_lstm_infer.py --image_path <path_to_image/image.jpg>
```

## The web page
We also deploy our models onto a ...
