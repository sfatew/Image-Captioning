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
* To run the model:
### Model Checkpoints
The model checkpoints are stored in the drive <>

### To run the model

put the models in the drive into the folder `/model`

You can run the following command to test on the image at your working directory

```
python3 infer.py --image_path path_to_image/image.jpeg --checkpoint model/model.pth
```
