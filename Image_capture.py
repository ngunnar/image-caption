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
        self.reshape = tf.keras.layers.Reshape((1, embedding_dim), input_shape = (embedding_dim, ))
    
    def call(self, x):
        x = self.cnn_model(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.reshape(x)
        return x

class Entire_model(tf.keras.Model):
    def __init__(self, image_shape, embedding_dim, lstm_units, top_k, max_length, **kwargs):
        super(Entire_model, self).__init__(**kwargs)
        vocab_size = top_k + 1
        self.max_length = max_length
        self.cnn_model = CNN_model(image_shape, embedding_dim)
        self.word_embedding = tf.keras.layers.Embedding(input_dim=top_k, output_dim=embedding_dim)
        # = tf.keras.layers.TimeDistributed()
        self.lstm_units = lstm_units
        self.lstm = tf.keras.layers.LSTM(self.lstm_units,
                                activation='linear',
                                recurrent_activation='sigmoid', 
                                return_sequences = True, 
                                return_state = True, 
                                stateful = False,
                                recurrent_initializer='glorot_uniform')
        self.lambda_ = tf.keras.layers.Lambda(lambda x: x[:, -1:, :])
        self.dense1 = tf.keras.layers.Dense(self.lstm_units)
        self.dense2 = tf.keras.layers.Dense(vocab_size)
        self.softmax = tf.keras.layers.Softmax()
    
    def reset_state(self, batch_size):
        return [tf.zeros((batch_size, self.lstm_units)), tf.zeros((batch_size, self.lstm_units))]
    '''
    def compile_special(self, optimizer, loss):
        #super(Entire_model, self).compile(**kwargs)
        self.optimizer = optimizer
        self.loss_fn = loss
        super().compile()
    def compile(self, **kwargs):
        raise NotImplementedError("Please use special_compile()")

    def call(self, x):
        img = x['input_1']
        word = x['input_2']
        f = self.cnn_model(img)
        dec_input = tf.expand_dims(cap[:, 0], 1)
        for i in range(1, cap.shape[1]):
            ws = model.word_embedding(dec_input)
            lstm_input = tf.concat([f, ws], axis=-1)
            if i == 1:    
                x, state_h, state_c = model.lstm(lstm_input, initial_state = initial_state)
            else:
                x, state_h, state_c = model.lstm(lstm_input)
            x = model.dense1(x)
            x = tf.reshape(x, (-1, x.shape[2]))
            x = model.dense2(x)
            dec_input = tf.expand_dims(cap[:, i], 1)
            
            #print("pred:", x.shape, target[:, i].shape)
            #loss += Entire_model.loss_function(cap[:, i], x)
            # using teacher forcing
            dec_input = tf.expand_dims(cap[:, i], 1)



        initial_state = self.reset_state(tf.shape(word)[0])
        img_feat = self.cnn_model(img)
        lstm_input = img_feat
        #_, state_h, state_c = self.lstm(lstm_input)
        #initial_state = [state_h, state_c]
        print(word.shape)
        ws = self.word_embedding(word)
        #x, _, _ = self.lstm(ws, initial_state)
        #print("X1", x.shape)
        #x = self.dense1(x)
        #print("X2", x.shape)
        #x = self.dense2(x)
        #print("X3", x.shape)
        #x = self.softmax(x)
        #print("X4", x.shape)
        
        for i in range(ws.shape[1]):
            lstm_input = tf.concat([img_feat, ws[:,i:i+1,...]], axis=-1)
            #print(lstm_input.shape)
            #lstm_input = ws[:,i:i+1,...]
            if i == 0:
                x, state_h, state_c = self.lstm(lstm_input, initial_state)
            else:
                x, state_h, state_c = self.lstm(lstm_input)
            initial_state = [state_h, state_c]
            #x = x[:,None,...]
            #print("X1", x.shape)
            x = self.lambda_(x)
            #print("X2", x.shape)
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.softmax(x)
            if i == 0:
                out = x
            else:
                out = tf.concat([out, x], axis=1)
        
        
        #_, x = tf.split(x, [1, self.max_length], 1)
        #print("OUT", out.shape)
        return out

    def train_step(self, data):
        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            t_loss = self.loss_fn(data['input_2'], y_pred)
        grads = tape.gradient(t_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': t_loss}
    '''
    def loss_function(real, pred):
        #print("loss", real, pred)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        #print("LOSS: ", loss_)
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
        def train_step(model, img_tensor, cap, tokenizer, batch_size):
            loss = 0
            initial_state = model.reset_state(cap.shape[0])
            #dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * cap.shape[0], 1)
            with tf.GradientTape() as tape:
                #lstm_input = model.cnn_model(img)
                #_, _, _ = model.lstm(lstm_input, initial_state = initial_state)
                #dec_input = tf.expand_dims(cap[:, 0], 1)
                for i in range(0, cap.shape[1]):
                    if i == 0:
                        lstm_input = model.cnn_model(img_tensor)
                        x, state_h, state_c = model.lstm(lstm_input, initial_state = initial_state)
                        initial_state = [state_h, state_c]
                        dec_input = tf.expand_dims(cap[:, i], 1)
                        continue
                    
                    #lstm_input = dec_input
                    ws = model.word_embedding(dec_input)
                    #lstm_input = tf.concat([f, ws], axis=-1)
                    lstm_input = ws
                    #if i == 1:    
                    #    x, state_h, state_c = model.lstm(lstm_input, initial_state = initial_state)
                    #else:
                    x, state_h, state_c = model.lstm(lstm_input, initial_state = initial_state)
                    initial_state = [state_h, state_c]
                    x = model.dense1(x)
                    x = tf.reshape(x, (-1, x.shape[2]))
                    x = model.dense2(x)
                    
                    #print("pred:", x.shape, target[:, i].shape)
                    loss += Entire_model.loss_function(cap[:, i], x)
                    # using teacher forcing
                    dec_input = tf.expand_dims(cap[:, i], 1)

                total_loss = (loss / int(cap.shape[1]))

                trainable_variables = model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))
                
            return loss, total_loss
        
        for epoch in range(start_epoch, epochs):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, cap)) in enumerate(dataset):
                #print(target.shape)
                batch_loss, t_loss = train_step(self, 
                                                img_tensor,
                                                cap,
                                                dataLoader.tokenizer,
                                                batch_size)
                total_loss += t_loss

                if batch % 100 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(cap.shape[1])))
            
            # storing the epoch end loss value to plot later
            self.loss_plot.append(total_loss / num_steps)

            #if epoch % 5 == 0:
            #    ckpt_manager.save()

            print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
            print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    
    def predict(self, img_tensor, tokenizer):
        initial_state = self.reset_state(batch_size=1)
        lstm_input = self.cnn_model(img_tensor)
        x, state_h, state_c = self.lstm(lstm_input, initial_state = initial_state)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []
        for i in range(self.max_length):
            ws = self.word_embedding(dec_input)
            lstm_input = ws#tf.concat([f, ws], axis=-1)
            #if i == 0:    
            x, state_h, state_c = self.lstm(lstm_input, initial_state = initial_state)
            #else:
            #x, state_h, state_c = self.lstm(lstm_input)
            initial_state = [state_h, state_c]
            x = self.dense1(x)
            x = tf.reshape(x, (-1, x.shape[2]))
            x = self.dense2(x)
            predicted_id = tf.random.categorical(x, 1)[0][0].numpy()
            if tokenizer.index_word[predicted_id] == '<end>':
                return result
            result.append(tokenizer.index_word[predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)
        return result
    
    '''
    def predict(self, img, tokenizer):
        img_feat = self.cnn_model(img)
        start_token = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        x = self.lstm(img_feat)
        for i in range(self.max_length):

            predicted_id = tf.random.categorical(x, 1)[0][0].numpy()
            if tokenizer.index_word[predicted_id] == '<end>':
                return result
            result.append(tokenizer.index_word[predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)
            return result
    '''

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