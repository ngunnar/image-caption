import tensorflow as tf
import time

def map_func(image_path, cap):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img, cap

class CNN_model(tf.keras.Model):
    def __init__(self, image_shape, embedding_dim):
        super(CNN_model, self).__init__()
        image_model = tf.keras.applications.VGG16(include_top=False,  weights='imagenet', input_shape=image_shape)
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output
        self.cnn_model = tf.keras.Model(new_input, hidden_layer)
        self.cnn_model.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')
    
    def call(self, x):
        x = self.cnn_model(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class RNN_model(tf.keras.Model):
    def __init__(self, top_k, embedding_dim, lstm_units, vocab_size):
        super(RNN_model, self).__init__()
        self.units = lstm_units
        self.word_embedding = tf.keras.layers.Embedding(input_dim=top_k, output_dim=embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.units,
                                activation='linear',
                                recurrent_activation='sigmoid', 
                                return_sequences = True, 
                                return_state = True, 
                                recurrent_initializer='glorot_uniform')
        self.dense1 = tf.keras.layers.Dense(self.units)
        self.dense2 = tf.keras.layers.Dense(vocab_size)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        S = x[0]
        features = x[1]
        if len(x) > 2:
            states = x[2]
        ws = self.word_embedding(S)
        #print("RNN ", S.shape, ws.shape, features.shape)
        lstm_input = tf.concat([tf.expand_dims(features,1), ws], axis=-1)
        if len(x) > 2:
            x, state_h, state_c = self.lstm(lstm_input, initial_state = states)
        else:
            x, state_h, state_c = self.lstm(lstm_input)
        #print("LSTM: ", x.shape)
        x = self.dense1(x)
        #print("Dense1: ", x.shape)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.dense2(x)
        #x = self.softmax(x)
        #print("OUT: ", x.shape)
        return x, state_h, state_c

    def reset_state(self, batch_size):
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]

class Model(object):

    def __init__(self, image_shape=(299,299,3), lstm_units=512, embedding_dim = 256, top_k=5000, max_length=80):
        self.image_shape = image_shape
        self.lstm_units = lstm_units
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.vocab_size = top_k + 1
        self.max_length = max_length
        self.cnn_model = CNN_model(self.image_shape, self.embedding_dim)
        self.rnn_model = RNN_model(self.top_k, self.embedding_dim, self.lstm_units, self.vocab_size)
    
    def loss_function(real, pred):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
    
    def train(self, dataLoader, batch_size, buffer_size, epochs=10):
        num_steps = len(dataLoader.img_name_train) // batch_size
        dataset = tf.data.Dataset.from_tensor_slices((dataLoader.img_name_train, dataLoader.cap_train))
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.loss_plot = []
        optimizer = tf.keras.optimizers.Adam()
        start_epoch = 0
        
        @tf.function
        def train_step(cnn_model, rnn_model, img_tensor, target, tokenizer, batch_size):
            loss = 0
            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            initial_state = rnn_model.reset_state(batch_size)
            dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

            with tf.GradientTape() as tape:
                f = cnn_model(img_tensor)

                for i in range(1, target.shape[1]):
                    #print(dec_input.shape, f.shape, [in_s.shape for in_s in initial_state])
                    if i == 1:
                        x, state_h, state_c = rnn_model([dec_input, f, initial_state])
                    else:
                        x, state_h, state_c = rnn_model([dec_input, f])
                    initial_state = [state_h, state_c]#[state_c, state_h]
                    
                    #print("pred:", x.shape, target[:, i].shape)
                    loss += Model.loss_function(target[:, i], x)
                    # using teacher forcing
                    dec_input = tf.expand_dims(target[:, i], 1)

                total_loss = (loss / int(target.shape[1]))

                trainable_variables = cnn_model.trainable_variables + rnn_model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))
            return loss, total_loss
        
        for epoch in range(start_epoch, epochs):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(dataset):
                #print(target.shape)
                batch_loss, t_loss = train_step(self.cnn_model, 
                                                      self.rnn_model, 
                                                      img_tensor, 
                                                      target, 
                                                      dataLoader.tokenizer,
                                                      batch_size)
                total_loss += t_loss

                if batch % 100 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
            
            # storing the epoch end loss value to plot later
            self.loss_plot.append(total_loss / num_steps)

            #if epoch % 5 == 0:
            #    ckpt_manager.save()

            print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
            print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def predict(self, img_tensor, tokenizer):
        initial_state = self.rnn_model.reset_state(batch_size=1)
        f = self.cnn_model(img_tensor)
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.max_length):
            if i == 0:
                x, state_h, state_c = self.rnn_model([dec_input, f, initial_state])
            else: 
                x, state_h, state_c = self.rnn_model([dec_input, f])
            initial_state = [state_h, state_c]#[state_c, state_h]
            predicted_id = tf.random.categorical(x, 1)[0][0].numpy()
            if tokenizer.index_word[predicted_id] == '<end>':
                return result
            result.append(tokenizer.index_word[predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)
        return result