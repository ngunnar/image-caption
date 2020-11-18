from tensorflow.python.ops import summary_ops_v2
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import os, math, io, time
import tensorflow as tf

from .create_dataset import read_img

def Tensorboard_callback(log_dir, t_img_file, t_cap, v_img_file, v_cap, model, tokenizer):
    class CustomTensorBoard(TensorBoard):
        def __init__(self, **kwargs):  # add other arguments to __init__ if you need
            super().__init__(**kwargs)
        
        def on_epoch_end(self, epoch, logs={}):            
            
            def _get_img(img_tensor, img, cap):
                result1 = model.predict(img_tensor[None,...], tokenizer, s_type='sampling')
                result2 = model.predict(img_tensor[None,...], tokenizer, s_type='greedy')
                result = [' '.join(result1), ' '.join(result2)]
                fig = plt.figure()
                plt.imshow(img)
                plt.axis('off')                
                plt.title('\n'.join(result), fontsize=8) 
                fig.text(.5, -.05, '\n'.join(cap), ha='center', fontsize=7)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                # Closing the figure prevents it from being displayed directly inside
                # the notebook.
                plt.close()
                buf.seek(0)
                # Convert PNG buffer to TF image
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                # Add the batch dimension
                image = tf.expand_dims(image, 0)            
                return image
            
            img_train = _get_img(t_img_tensor, t_img, t_cap)
            if v_img_file is not None:
                img_val = _get_img(v_img_tensor, v_img, v_cap)
            with file_writer.as_default():
                tf.summary.image("Train_Prediction", img_train, step=epoch)
                if v_img_file is not None:
                    tf.summary.image("Val_Prediction", img_val, step=epoch)                        
                
            super().on_epoch_end(epoch, logs)
    
    
    t_img_tensor = read_img(t_img_file)    
    t_img = tf.io.read_file(t_img_file)
    t_img = tf.image.decode_jpeg(t_img, channels=3)
    
    if v_img_file is not None:
        v_img_tensor = read_img(v_img_file)
        v_img = tf.io.read_file(v_img_file)
        v_img = tf.image.decode_jpeg(v_img, channels=3)
    
    # Tensorboard
    file_writer = tf.summary.create_file_writer(log_dir + '/img')
    return CustomTensorBoard(log_dir= log_dir, 
                       histogram_freq=1,
                       profile_batch = 0,
                       embeddings_freq=0,
                       write_grads=False)