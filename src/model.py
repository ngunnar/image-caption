import tensorflow as tf
import time

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    loss_ = tf.reduce_mean(loss_)
    return tf.reduce_mean(loss_)

class CNN_model(tf.keras.Model):
    def __init__(self, image_shape, embedding_dim, name):
        super(CNN_model, self).__init__(name=name)
        #image_model = tf.keras.applications.VGG16(include_top=False,  weights='imagenet', input_shape=image_shape)
        image_model = tf.keras.applications.InceptionV3(weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-2].output #-1
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

class model(tf.keras.Model):
    def __init__(self, image_shape, embedding_dim, lstm_units, top_k, max_length, embedding_matrix=None, **kwargs):
        super(model, self).__init__(**kwargs)
        vocab_size = top_k + 1
        self.max_length = max_length
        self.cnn_model = CNN_model(image_shape, embedding_dim, 'cnn_model')
        #self.cnn_dropout = tf.keras.layers.Dropout(0.5, name='cnn_dropout')
        if embedding_matrix is None:
            self.word_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, 
                                                            output_dim=embedding_dim,
                                                            mask_zero=False,
                                                            name='word_embedding')
        else:
            self.word_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, 
                                                            output_dim=embedding_dim,
                                                            weights=[embedding_matrix],
                                                            trainable=False,
                                                            mask_zero=False,
                                                            name='word_embedding')
        #self.embedding_dropout = tf.keras.layers.Dropout(0.5, name='embedding_dropout')
        self.lstm_units = lstm_units
        self.lstm = tf.keras.layers.LSTM(self.lstm_units,
                                         activation='tanh', 
                                         recurrent_activation='sigmoid',
                                         return_sequences=True,
                                         return_state = True,
                                         name='lstm')
        self.dense1 = tf.keras.layers.Dense(self.lstm_units, name='dense1')
        self.dense2 = tf.keras.layers.Dense(vocab_size, name='dense2')
        self.softmax = tf.keras.layers.Softmax(name='softmax')
        self.loss_fn = loss_function
    
    def call(self, inputs, training=False):
        img_tensor = inputs[0]
        cap_in = inputs[1]
        initial_state = self.reset_state(tf.shape(img_tensor)[0])
        img_feat = self.cnn_model(img_tensor) # Image feature
        if training:
            img_feat = self.cnn_dropout(img_feat)
        ws = self.word_embedding(cap_in)
        if training:
            ws = self.embedding_dropout(ws)
        lstm_input = tf.concat([img_feat, ws], axis=1)
        x, _, _ = self.lstm(lstm_input, initial_state = initial_state)
        _, x = tf.split(x, [1, self.max_length - 1], axis=-2)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x
    
    
    def reset_state(self, batch_size):
        return [tf.zeros((batch_size, self.lstm_units)), tf.zeros((batch_size, self.lstm_units))]
    
    '''
    def train(self, dataLoader, batch_size, buffer_size, epochs=10):
        num_steps = len(dataLoader.img_name_train) // batch_size
        dataset = create_dataset(dataLoader, batch_size, buffer_size)        
        self.loss_plot = []
        optimizer = tf.keras.optimizers.Adam()
        start_epoch = 0

        @tf.function
        def train_step(model, img_tensor, cap_in, cap_out, tokenizer):
            loss = 0
            initial_state = model.reset_state(img_tensor.shape[0])
            with tf.GradientTape() as tape:
                img_feat = model.cnn_model(img_tensor) # Image feature
                img_feat = model.cnn_dropout(img_feat)
                ws = model.word_embedding(cap_in)
                ws = model.embedding_dropout(ws)
                lstm_input = tf.concat([img_feat, ws], axis=1)
                x, _, _ = model.lstm(lstm_input, initial_state = initial_state)
                _, x = tf.split(x, [1, model.max_length - 1], axis=-2)
                x = model.dense1(x)
                x = model.dense2(x)
                x = model.softmax(x)
                loss = model.loss_fn(cap_out, x)
                
                total_loss = (loss / int(cap_out.shape[1]))

                trainable_variables = model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))
            return loss, total_loss
        
        @tf.function
        def val_step(model, img_tensor, cap_in, cap_out, tokenizer):
            loss = 0
            initial_state = model.reset_state(img_tensor.shape[0])
            img_feat = model.cnn_model(img_tensor) # Image feature
            img_feat = model.cnn_dropout(img_feat)
            ws = model.word_embedding(cap_in)
            ws = model.embedding_dropout(ws)
            lstm_input = tf.concat([img_feat, ws], axis=1)
            x, _, _ = model.lstm(lstm_input, initial_state = initial_state)
            _, x = tf.split(x, [1, model.max_length - 1], axis=-2)
            x = model.dense1(x)
            x = model.dense2(x)
            x = model.softmax(x)
            loss = model.loss_fn(cap_out, x)

            total_loss = (loss / int(cap_out.shape[1]))
            return loss, total_loss
        
        for epoch in range(start_epoch, epochs):
            start = time.time()
            total_loss = 0

            for (batch, (inputs, output)) in enumerate(dataset):
                #print("Batch {0}".format(batch))
                img_tensor = inputs[0]
                cap_in = inputs[1]
                cap_out = output
                batch_loss, t_loss = train_step(self, img_tensor, cap_in, cap_out, dataLoader.tokenizer)
                total_loss += t_loss

                if batch % 100 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, batch_loss.numpy() / int(cap_out.shape[1])))
            
            # storing the epoch end loss value to plot later
            self.loss_plot.append(total_loss / num_steps)

            #if epoch % 5 == 0:
            #    ckpt_manager.save()

            print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
            print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    '''
    
    # greedy decoder
    
    '''
    def greedy_decoder(pred):
        # index for largest probability each row
        return np.argmax(pred)
    
    def beam_search(data, k):
    ''' 
    
    def predict(self, img_tensor, tokenizer, s_type='sampling'):
        assert img_tensor.shape[0] == 1, "Only support prediction for one image a the moment"
        result = []
        bs = img_tensor.shape[0]
        img_feat = self.cnn_model(img_tensor)
        cap_pred = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        initial_state = self.reset_state(bs)
        import numpy as np
        greedy = False
        o = []
        for _ in range(self.max_length):
            ws = self.word_embedding(cap_pred)
            lstm_input = tf.concat([img_feat, ws], axis=1)
            x, state_h, state_c = self.lstm(lstm_input, initial_state)
            initial_state = [state_h, state_c]
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.softmax(x)
            o.append(x)
            if s_type == 'greedy':
                predicted_id = np.argmax(x[0,-1,:].numpy())
            else:
                predicted_id = tf.random.categorical(tf.math.log(x[:,-1,:]), 1)[0][0].numpy() #Sample, TODO BeamSearch?
            w = tokenizer.index_word[predicted_id]
            if w == '<end>':
                return result
            result.append(w)
            cap_pred = tf.concat([cap_pred, predicted_id[None,None,...]], axis=1)

        return result