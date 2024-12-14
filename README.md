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

```
python3 transformer_model_infer.py --image_path path_to_image/image.jpeg 
python3 transformer_model_pre_infer.py --image_path path_to_image/image.jpeg 
```

## The web page
We also deploy our models onto a ...
