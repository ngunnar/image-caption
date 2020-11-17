from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import os, math, io, time
import tensorflow as tf

from .create_dataset import read_img

def Tensorboard_callback(log_dir, t_img_file, v_img_file, model, tokenizer):
    class CustomTensorBoard(TensorBoard):
        def __init__(self, **kwargs):  # add other arguments to __init__ if you need
            super().__init__(**kwargs)
        
        def on_epoch_end(self, epoch, logs={}):            
            
            def _get_img(img_tensor, img):
                result = model.predict(img_tensor[None,...], tokenizer)
                plt.imshow(img)
                plt.axis('off')
                plt.title(' '.join(result))                             
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                # Closing the figure prevents it from being displayed directly inside
                # the notebook.
                plt.close()
                buf.seek(0)
                # Convert PNG buffer to TF image
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                # Add the batch dimension
                image = tf.expand_dims(image, 0)            
                return image, img
            
            img_train = _get_img(t_img_tensor, t_img)
            img_val = _get_img(v_img_tensor, v_img)
            return img_train
            #print(img_train)
            with file_writer.as_default():
                tf.summary.image("Train_Prediction", img_train, step=epoch)
                tf.summary.image("Val_Prediction", img_val, step=epoch)                        
                
            super().on_epoch_end(epoch, logs)
    
    
    t_img_tensor = read_img(t_img_file)
    v_img_tensor = read_img(v_img_file)
    
    t_img = tf.io.read_file(t_img_file)
    t_img = tf.image.decode_jpeg(t_img, channels=3)
    
    v_img = tf.io.read_file(v_img_file)
    v_img = tf.image.decode_jpeg(v_img, channels=3)
    
    # Tensorboard
    file_writer = tf.summary.create_file_writer(log_dir + '/img')
    return CustomTensorBoard(log_dir= log_dir, 
                       histogram_freq=1,
                       profile_batch = 0,
                       embeddings_freq=0,
                       write_grads=False)