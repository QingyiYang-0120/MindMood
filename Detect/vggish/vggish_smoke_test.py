
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import sys
sys.path.append('E:/ICASSP2022-Depression-main/')
from vggish import vggish_input
from vggish import vggish_params
from vggish import vggish_postprocess
from vggish import vggish_slim

print('\nTesting your install of VGGish\n')
# In[]
# Paths to downloaded VGGish files.
checkpoint_path = 'E:/ICASSP2022-Depression-main/vggish/vggish_model.ckpt'
pca_params_path = 'E:/ICASSP2022-Depression-main/vggish/vggish_pca_params.npz'

# Relative tolerance of errors in mean and standard deviation of embeddings.
rel_error = 0.1  # Up to 10%

# Generate a 1 kHz sine wave at 44.1 kHz (we use a high sampling rate
# to test resampling to 16 kHz during feature extraction).
num_secs = 3
freq = 1000
sr = 44100
t = np.arange(0, num_secs, 1 / sr)
x = np.sin(2 * np.pi * freq * t)

# Produce a batch of log mel spectrogram examples.
input_batch = vggish_input.waveform_to_examples(x, sr)
print('Log Mel Spectrogram example: ', input_batch[0])
np.testing.assert_equal(
    input_batch.shape,
    [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

# Define VGGish, load the checkpoint, and run the batch through the model to
# produce embeddings.
with tf.Graph().as_default(), tf.Session() as sess:
  vggish_slim.define_vggish_slim()
  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

  features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)
  [embedding_batch] = sess.run([embedding_tensor],
                               feed_dict={features_tensor: input_batch})
  print('VGGish embedding: ', embedding_batch[0])
  expected_embedding_mean = -0.0333
  expected_embedding_std = 0.380
  np.testing.assert_allclose(
      [np.mean(embedding_batch), np.std(embedding_batch)],
      [expected_embedding_mean, expected_embedding_std],
      rtol=rel_error)

# Postprocess the results to produce whitened quantized embeddings.
pproc = vggish_postprocess.Postprocessor(pca_params_path)
postprocessed_batch = pproc.postprocess(embedding_batch)
print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
expected_postprocessed_mean = 122.0
expected_postprocessed_std = 93.5
np.testing.assert_allclose(
    [np.mean(postprocessed_batch), np.std(postprocessed_batch)],
    [expected_postprocessed_mean, expected_postprocessed_std],
    rtol=rel_error)

print('\nLooks Good To Me!\n')