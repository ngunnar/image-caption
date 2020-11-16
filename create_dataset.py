import tensorflow as tf

def read_img(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def input_generator(img_name_data, cap_data):
    def gen():
        for img, cap in zip(img_name_data, cap_data):
            img = read_img(img)
            cap = tf.convert_to_tensor(cap)
            yield img, cap[:-1]
    return gen

def output_generator(cap_data):
    def gen():
        for cap in cap_data:
            #cap = tf.convert_to_tensor(cap)
            yield cap[1:]
    return gen

def create_dataset(img_name_data, cap_data, batch_size, buffer_size):
    #steps_per_epoch = int(len(dataLoader.cap_train)/batch_size)
    #print("Number of steps per epoch {0}".format(steps_per_epoch))
    input_data = tf.data.Dataset.from_generator(
                input_generator(img_name_data, cap_data),
                output_types=(tf.float32, tf.float64),
                output_shapes= ((None,None,None), (None,)))

    output_data = tf.data.Dataset.from_generator(
                output_generator(cap_data),
                output_types= tf.float64,
                output_shapes= ((None,)))
    dataset = tf.data.Dataset.zip((input_data, output_data))
            
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset