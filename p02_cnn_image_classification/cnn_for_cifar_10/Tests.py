from p02_cnn_image_classification.cnn_for_cifar_10.ensemble_cnn_model_test import Model
import numpy as np
import tensorflow as tf
import time


# training_epochs = 20
batch_size = 100

train_file_list = ['data/train_data_' + str(i) + '.csv' for i in range(1, 51)]
test_file_list = ['data/test_data_' + str(i) + '.csv' for i in range(1, 11)]

def data_setting(data):
    # x : 데이터, y : 라벨
    x = (np.array(data[:, 0:-1]) / 255).tolist()
    y_tmp = np.zeros([len(data), 10])
    for i in range(0, len(data)):
        label = int(data[i][-1])
        y_tmp[i, label - 1] = 1
    y = y_tmp.tolist()

    return x, y


def read_data(filename):
# def read_data(*filename):
    ####################################################################################################################
    ## ▣ Data Loading
    ##  - 각각의 파일에 대해 load 후 전처리를 수행
    ####################################################################################################################
    #data1 = np.loadtxt(filename[0], delimiter=',')
    #data2 = np.loadtxt(filename[1], delimiter=',')
    #data = np.append(data1, data2, axis=0)
    data = np.loadtxt(filename, delimiter=',')
    np.random.shuffle(data)
    return data_setting(data)

tf.reset_default_graph()

########################################################################################################################
## ▣ Data Test
##  - test data : 10,000 개
##  - batch_size : 100, model : 5개
########################################################################################################################
with tf.Session() as sess:
    models = []
    num_models = 5
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, 'log/epoch_' + str(23) + '.ckpt')

    print('Testing Started!')

    ensemble_accuracy = 0.
    model_accuracy = [0., 0., 0., 0., 0.]
    cnt = 0
    ensemble_confusion_mat = np.zeros((10, 10))

    for index in range(0, len(test_file_list)):
        total_x, total_y = read_data(test_file_list[index])
        for start_idx in range(0, 1000, batch_size):
            test_x_batch, test_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
            test_size = len(test_y_batch)
            predictions = np.zeros(test_size * 10).reshape(test_size, 10)

            model_result = np.zeros(test_size*10, dtype=np.int).reshape(test_size, 10)
            model_result[:, 0] = range(0, test_size)

            for idx, m in enumerate(models):
                model_accuracy[idx] += m.get_accuracy(test_x_batch, test_y_batch)
                p = m.predict(test_x_batch)
                model_result[:, 1] = np.argmax(p, 1)
                for result in model_result:
                    predictions[result[0], result[1]] += 1

            ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y_batch, 1))
            ensemble_accuracy += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
            ensemble_confusion_mat = tf.add(tf.contrib.metrics.confusion_matrix(labels=tf.argmax(test_y_batch, 1),
                                                                                predictions=tf.argmax(predictions, 1),
                                                                                num_classes=10, dtype='int32',
                                                                                name='confusion_matrix'),
                                            ensemble_confusion_mat)
            cnt += 1
    for i in range(len(model_accuracy)):
        print('Model ' + str(i) + ' : ', model_accuracy[i] / cnt)
    print('Ensemble Accuracy : ', sess.run(ensemble_accuracy) / cnt)
    print('Testing Finished!')
    print('####### Confusion Matrix #######')
    print(sess.run(ensemble_confusion_mat))
    # print(sess.run(tf.contrib.metrics.confusion_matrix(labels=tf.arg_max(total_y, dimension=1), predictions=tf.arg_max(m.predict(total_x), dimension=1), num_classes=10, dtype='int32', name='confusion_matrix')))
