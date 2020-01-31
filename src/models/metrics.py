import tensorflow.keras.backend as K

def dice_coeff(y_true, y_pred, smooth=1):

    y_true_f = K.cast(
        K.greater(K.flatten(y_true), 0.5), 'float32')
    y_pred_f = K.cast(
        K.greater(K.flatten(y_pred), 0.5), 'float32')

    a_inter_b = K.sum(y_true_f * y_pred_f)

    score = ((2. * a_inter_b) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score