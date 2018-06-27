from keras import backend as k
import tensorflow as tf

lambda_coefficient = 0.1


def doc_total_loss(y_true, y_pred):
    discriminative_loss = y_true[0][0]

    mean = k.mean(y_pred, axis=0)
    diff_step = y_pred - mean
    sample_variance = k.sum(k.square(diff_step), axis=1)
    var_sum = k.sum(sample_variance)

    compactness_loss = tf.cast((tf.shape(y_pred)[0] / (tf.shape(y_pred)[1] * k.pow(tf.shape(y_pred)[0] - 1, 2))),
                               tf.float32) * var_sum

    total_loss = discriminative_loss + (lambda_coefficient * compactness_loss)

    return total_loss
