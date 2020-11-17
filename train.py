import argparse
import tensorflow as tf
import os
import datetime

from src.model import model, loss_function
from src.create_dataset import create_dataset
from src.dataLoader import DataLoader
from src.CustomTensorboard import Tensorboard_callback

def main(top_k, data_path, ds_size, embedding_dim, epochs, lr, batch_size):
    dataLoader = DataLoader(data_path, ds_size)
    dataLoader.convert_to_dataset(top_k)
    if len(dataLoader.tokenizer.index_word) < top_k:
        top_k = len(dataLoader.tokenizer.index_word) - 1 # If our dictonary is less than top_k. -1 because <unk> is included
        dataLoader.top_k = top_k
    max_length = dataLoader.max_length
    
    print("Training size: {0}, images: {1}".format(len(dataLoader.img_name_train), len(set(dataLoader.img_name_train))))
    print("Validation size: {0}, images: {1}".format(len(dataLoader.img_name_val), len(set(dataLoader.img_name_val))))
    print("Max length: {0}".format(max_length))
    print("Top k: {0}".format(top_k))
    print("Epochs: {0}, batch size (per replica): {1}".format(epochs, batch_size)) 
    
    image_shape = (299,299,3)
    embedding_matrix = None
    lstm_units = embedding_dim

    buffer_size = 1000
    train_dataset = create_dataset(dataLoader.img_name_train, dataLoader.cap_train, batch_size, buffer_size)
    val_dataset = create_dataset(dataLoader.img_name_val, dataLoader.cap_val, batch_size, buffer_size)
    num_batches = int(len(dataLoader.cap_train)/batch_size)
    
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "./logs/fit/" + dt
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./Saved_models/best_{0}'.format(dt),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    batch_size *= mirrored_strategy.num_replicas_in_sync
    train_batches = int(len(dataLoader.cap_train)/batch_size)
    val_batches = int(len(dataLoader.cap_val)/batch_size)
    with mirrored_strategy.scope():
        m = model(image_shape, embedding_dim, lstm_units, top_k, max_length)
        tensorboard_callback = Tensorboard_callback(log_dir=log_dir,
                                             t_img_file=dataLoader.img_name_train[0],
                                             v_img_file=dataLoader.img_name_val[0],
                                             model=m,
                                             tokenizer=dataLoader.tokenizer)
        m.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), loss = loss_function)
        history = m.fit(train_dataset, 
                        epochs = epochs,
                        steps_per_epoch = train_batches,
                        shuffle=True,
                        callbacks = [tensorboard_callback, model_checkpoint_callback],
                        validation_data = val_dataset,
                        validation_steps = val_batches)
        m.save_weights('./Saved_models/{0}'.format(dt))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", help="Dataset root", type=str, required=True) #'/data/Niklas/Flickr30k'
    parser.add_argument("-ds_size", help="Dataset size, 0 = use all", type=int, default=0)
    parser.add_argument("-top_k", help="Use top k words", type=int, default=3000)
    parser.add_argument("-epochs", help="Number of epochs", type=int, default=50)
    parser.add_argument("-batch_size", help="Batch size", type=int, default=16)
    parser.add_argument("-embedding_dim", help="Embedding dimension", type=int, default=512)
    parser.add_argument("-lr", help="Learning rate", type=float, default=0.001)
    parser.add_argument("-gpus", help="use gpus, example -1 (for cpu), 0 for gpu 0, 0,1,2 for gpu 0,1,2", type=str, default="1,2,3,4")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    dataset_root = args.ds
    ds_size = args.ds_size
    top_k = args.top_k
    embedding_dim = args.embedding_dim
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    
    main(top_k, dataset_root, ds_size, embedding_dim, epochs, lr, batch_size)