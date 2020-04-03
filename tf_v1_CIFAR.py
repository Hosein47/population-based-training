
import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper
from utils import FloatTensorLike, TensorLike, AcceptableDTypes


from typeguard import typechecked
from typing import Union, Optional


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):

    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( tf.where(y_true[i])[0] )
        set_pred = set( tf.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(tf.set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def hamming_loss_fn(
    y_true: TensorLike,
    y_pred: TensorLike,
    threshold: Union[FloatTensorLike, None],
    mode: str,
) -> tf.Tensor:

    if mode not in ["multiclass", "multilabel"]:
        raise TypeError("mode must be either multiclass or multilabel]")

    if threshold is None:
        threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
        # make sure [0, 0, 0] doesn't become [1, 1, 1]
        # Use abs(x) > eps, instead of x != 0 to check for zero
        y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
    else:
        y_pred = y_pred > threshold

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)

    if mode == "multiclass":
        nonzero = tf.cast(tf.math.count_nonzero(y_true * y_pred, axis=-1), tf.float32)
        return 1.0 - nonzero

    else:
        nonzero = tf.cast(tf.math.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
        return nonzero / y_true.get_shape()[-1]


class HammingLoss(MeanMetricWrapper):
    """Computes hamming loss."""

    @typechecked
    def __init__(
        self,
        mode: str,
        name: str = "hamming_loss",
        threshold: Optional[FloatTensorLike] = None,
        dtype: AcceptableDTypes = None,
        **kwargs
    ):
        super().__init__(
            hamming_loss_fn, name=name, dtype=dtype, mode=mode, threshold=threshold
        )
