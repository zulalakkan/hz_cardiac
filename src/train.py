import os
import tensorflow as tf
import matplotlib.pyplot as plt
from network import build_ResUNet
from data_generator import DataGenerator

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('image_size',192,
                           'Image size')
tf.compat.v1.flags.DEFINE_integer('batch_size',5,
                           'Batch size')
tf.compat.v1.flags.DEFINE_integer('epochs',100,
                           'Number of epochs')
tf.compat.v1.flags.DEFINE_integer('num_class', 4,
                           'Number of segmentation classes')                            
tf.compat.v1.flags.DEFINE_string('data_dir', 'C:/Users/ilkay/Desktop/CMR Segm/M&Ms',
                           'Data directory')
tf.compat.v1.flags.DEFINE_string('train_csv', '',
                           'Train csv')

def get_data_ids(data_type): #training or validation
    data_dir = os.path.join(FLAGS.data_dir, data_type)
    data_ids = []
    for data in os.listdir(data_dir):
        data_ids.append((data, "ED"))
        data_ids.append((data, "ES"))

    return data_ids

if __name__ == '__main__':
    train_ids = get_data_ids("Train")
    valid_ids = get_data_ids("Validation")

    train_dir = os.path.join(FLAGS.data_dir, "Train")
    valid_dir = os.path.join(FLAGS.data_dir, "Validation")

    train_gen = DataGenerator(train_ids, train_dir, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size)
    print("***************", train_gen.__getitem__(0)[1].shape)
    valid_gen = DataGenerator(valid_ids, valid_dir, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size)

    train_steps = len(train_ids)//FLAGS.batch_size
    valid_steps = len(valid_ids)//FLAGS.batch_size

    epochs = FLAGS.epochs

    model = build_ResUNet(FLAGS.image_size, FLAGS.num_class)

    history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=epochs)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    
    model.save("ResUNet2.h5")