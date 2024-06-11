
# This test code was derived from:
# https://github.com/andreped/GradientAccumulator/
# with slight modification for supporting TF2.16 (and Keras 2.16)

import os
import random as python_random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tf_keras as keras
from tf_keras.models import load_model

from ga import GAOptimizer


# get current tf minor version
tf_version = int(tf.version.VERSION.split(".")[1])

Custom_Objects = {
  "GAOptimizer": GAOptimizer,
}


def run_experiment(
    bs=100, accum_steps=1, epochs=1, strategy_name="multi"
):
    # setup single/multi-GPU strategy
    if strategy_name == "single":
        strategy = tf.distribute.get_strategy()  # get default strategy
    elif strategy_name == "multi":
        cross_device_ops = tf.distribute.NcclAllReduce(num_packs=1)
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
    else:
        raise ValueError("Unknown distributed strategy chosen:", strategy_name)

    # load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # build train pipeline
    ds_train = ds_train.map(normalize_img)
    ds_train = ds_train.batch(bs)
    ds_train = ds_train.prefetch(1)

    # build test pipeline
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(bs)
    ds_test = ds_test.prefetch(1)

    with strategy.scope():
        # create model
        model = keras.models.Sequential(
            [
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(10),
            ]
        )

        # define optimizer - currently only SGD compatible with GAOptimizerWrapper
        opt = keras.optimizers.Adam(learning_rate=1e-3)

        # wrap optimizer to add gradient accumulation support
        if accum_steps >= 2:
            opt = GAOptimizer(optimizer=opt, accum_steps=accum_steps)

        # compile model
        model.compile(
            optimizer=opt,
            loss=keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

    # train model
    model.fit(
        ds_train,
        batch_size=bs,
        epochs=epochs,
        validation_data=ds_test,
        verbose=1,
    )

    model.save("./trained_model")

    # load trained model and test
    del model
    # trained_model = load_model("./trained_model", custom_objects=Custom_Objects, compile=True)
    trained_model = load_model("./trained_model", compile=True)

    del strategy

    result = trained_model.evaluate(ds_test, verbose=1)
    print(result)
    return result[1]


def test_distributed_optimizer_invariance():
    # use mixed precision
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)

    # run experiment for different optimizers, to see if GA is consistent
    # within an optimizer. Note that it is expected for the results to
    # differ BETWEEN optimizers, as they behave differently.
    for strategy_name in ["single", "multi"]:
        print("Current strategy: " + strategy_name)
        # set seed
        reset()

        # run once
        result1 = run_experiment(
            bs=100,
            accum_steps=1,
            epochs=2,
            strategy_name=strategy_name,
        )

        # reset before second run to get identical results
        reset()

        # run again with different batch size and number of accumulations
        result2 = run_experiment(
            bs=50,
            accum_steps=2,
            epochs=2,
            strategy_name=strategy_name,
        )

        # results should be "identical" (on CPU, can be different on GPU)
        np.testing.assert_almost_equal(result1, result2, decimal=2)


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255.0, label

def reset(seed=123):
  # set tf log level
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

  # disable GPU
  # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

  # clear keras session
  keras.backend.clear_session()

  os.environ["PYTHONHASHSEED"] = str(seed)

  # The below is necessary for starting Numpy generated random numbers
  # in a well-defined initial state.
  np.random.seed(seed)

  # The below is necessary for starting core Python generated random numbers
  # in a well-defined state.
  python_random.seed(seed)

  # The below set_seed() will make random number generation
  # in the TensorFlow backend have a well-defined initial state.
  # For further details, see:
  # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
  # @TODO: Should this seed be different than for python and numpy?
  tf.random.set_seed(seed)

  # https://stackoverflow.com/a/71311207
  try:
    tf.config.experimental.enable_op_determinism()  # Exist only for TF > 2.7
  except AttributeError as e:
    print(e)

  # force cpu threading determinism
  # https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed
  tf.config.threading.set_inter_op_parallelism_threads(1)
  tf.config.threading.set_intra_op_parallelism_threads(1)

if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    test_distributed_optimizer_invariance()
