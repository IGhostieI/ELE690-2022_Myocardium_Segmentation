# ELE690-2022: Myocardium Segmentation

## Image Extraction <a name="image extraction"/>
*get_images_and_masks.py* is used to extract images and Mmyocardium masks. In order to use this function it is currently required to have the files from [CardioMiner](https://github.com/Biomedical-Data-Analysis-Laboratory/CardioMiner). The variable *filepath* and *filepathDel* must both be used within the UNIX-server provided by the University of Stavanger. The extraction process takes between 10 and 30 min. To avoid running the exctraction process when using the haglag- or vxvy dataset some pickle files can be found at "/home/prosjekt5/BMDLab/users/casperc/". 

### Crop Heart
Within *CropHeart.py* is a function for cropping the image and myocardium mask to an area closer around the myocardium to reduce the amount of backgorund pixels. It is based on Kjersti Engan's matlab function, and translated into Python. 

## Training and Evaluation of Model
*Myocardium_segmentation_full.ipynb* is a jupyter notebook containing a program to train and save a UNet model for myocardium segmentation. The first cell after import 
cell is preprocessing of the data. The data is reshaped into a format that tensorflow can interpret. The numy arrays from the pickle files have the shape (x, 256, 256), while tensorflow needs the shape to have a dedicated channel dimension. The model input must have input shape (x, 128, 128, 1). The model is made with *create_unet_model()*. In order to actually run train the model set train = True.

## Prediction and plotting of masks on pretrained model
The weights for the models trained during the project is provided in the folder *h5*. The best-performing models provided are *val_accuracy.h5* and *full300.h5*. *prediction_and_plotting.py* can be used to predict a myocardium mask on a pretrained model. The functions *predict_myocardium()* and *plot_myocardium_mask()* are extracted from *pred_and_plot()* from *Myocardium_segmentation_full.ipynb*, and modified to work independently from each other. In the *main()* function an image is read using *cv2.imread()*, but pydicom can also be used to extract the pixel_array. *predict_myocardium()* takes the UNet model with pretrained weights and an image as input and returns a tuple containing a predicted mask, and a thresholded mask. *plot_myocardium_mask()* displays the input image with an overlayed myocardium mask.
