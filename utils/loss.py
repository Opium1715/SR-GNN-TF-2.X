import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


class Loss_with_L2(tf.keras.losses.Loss):

    def __init__(self, model, reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)
        self.model = model

    def call(self, y_true, y_pred):
        scc_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        variables_l2 = [tf.keras.regularizers.l2(l2=1e-5)(x) for x in self.model.trainable_variables]
        l2_loss = tf.add_n(variables_l2)
        return scc_loss + l2_loss
