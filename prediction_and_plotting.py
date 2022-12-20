import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, initializers
import numpy as np
import cv2


def create_unet_model(input_shape):
    #Buildeing an unet model
    inputs = layers.Input(input_shape)

    #Contraction path
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(c5)

    #Expansive path 
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(), padding='same')(c9)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs], name = "UNet_Model")

    return model
def predict_myocardium(unet_model, image):
    """
    Predict the myocardium mask based on a model with pretrained weights. 
    
    ## Inputs:
    - unet_model: A UNet model made with create_unet_model( )
    - image: An NDArray of shape (128, 128, 1)
    
    ## Outputs:
    Returns a tuple containting the predicted mask and the thresholded version of the prediction (pred, thresh)
    """
    image = np.array([image])
    preds_test = unet_model.predict(image,batch_size=1, verbose=0)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    return preds_test[0], preds_test_t[0]

def plot_myocardium_mask(predicted_Mmyo, image, save_path=None, ground_truth_Mmyo=None):
    """
    This function creates a figure with the predicted myocardium mask overlayed on the image. If a ground truth mask is provided two images are plotted with descriptive titles indicating which mask is predicted. If save_path is provided the image is saved to the specified location.

    ## Inputs:
    - predicted_Mmyo: A numpy array of shape (128, 128, 1) containing predicted myocardium masks with 1=myocardium, 0=background.
    - image: Image formatted as a numpy array of shape (128, 128, 1)
    - save_path: A string containing the save location and filename. ex. "images\\prediction.png"
    - ground_truth_Mmyo: A numpy array of shape (128, 128, 1) containing gorund truth myocardium masks with labels 1=myocardium, 0=background.
    """
    fig = plt.figure(figsize=(30,12))
    alpha = 0.5  # Transparency factor
    rgb = cv2.cvtColor((image*255).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    print(rgb.shape)
    if ground_truth_Mmyo is not None:
        # Add contours from ground truth
        
        cont, _ = cv2.findContours(ground_truth_Mmyo.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_cont = cv2.drawContours(rgb.copy(), cont, -1,(0,255,79),-1)
        img_cont = cv2.addWeighted(img_cont, alpha,rgb.copy() , 1 - alpha, 0)
        img_cont = cv2.drawContours(img_cont,cont,-2,(255,0,0),1)
        plt.subplot(2,1,1)
        plt.title('Ground Truth Mask',fontsize=30)
        plt.imshow(img_cont)

        plt.subplot(2,1,2)
        plt.title(f'Predicted Mask',fontsize=30)
    # Add contours from predicted myocardium
    pred_cont, _ = cv2.findContours(predicted_Mmyo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_pred_cont = cv2.drawContours(rgb.copy(),pred_cont,-2,(0,255,79),-1)
    img_pred_cont = cv2.addWeighted(img_pred_cont, alpha,rgb.copy() , 1 - alpha, 0)
    img_pred_cont = cv2.drawContours(img_pred_cont,pred_cont,-1,(255,0,0),1)
    plt.imshow(img_pred_cont)

    for ax in fig.axes:
        ax.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def main():
    # Define param
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 1
    image_path = '<path to image>'

    input_shape = (IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)

    # Load data
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
        
    resize_and_rescale = Sequential([
        layers.Resizing(IMG_WIDTH, IMG_HEIGHT),
        layers.Rescaling(1./255)
    
    ])

    processed_image = resize_and_rescale(image)
    processed_Mmyo = resize_and_rescale(image)

    model = create_unet_model(input_shape)
    model.load_weights("h5_files\val_accuracy.h5")

    _, Mmyo = predict_myocardium(unet_model=model, image=processed_image)
    plot_myocardium_mask(predicted_Mmyo=Mmyo, image=processed_image)
if __name__=="__main__":
    main()
