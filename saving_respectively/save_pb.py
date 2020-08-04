import os
import shutil
import tensorflow as tf
#importing GAN model is required

tf.compat.v1.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                                     """Directory where to read training checkpoints.""")
tf.compat.v1.app.flags.DEFINE_string('output_dir', './gan-export',
                                     """Directory where to export the model.""")
tf.compat.v1.app.flags.DEFINE_integer('model_version', 1,
                                      """Version Number of The Model""")
FLAGS=tf.compat.v1.app.flags.FLAGS

tf.compat.v1.disable_eager_execution()

#the input image preprocess is required
def preprocess_image(image_buffer):


    return image

def main(_):
    with tf.Graph(). as_default():
        serialized_tf_example = tf.compat.v1.placeholder(tf.string, name='input_image')
        feature_configs = {
            'image/encoded' : tf.io.FixedLenFeature(
                shape=[], dtype=tf.string),
        }
        tf_example=tf.io.parse_example(serialized_tf_example, feature_configs)
        jpegs = tf_example['image/encoded']
        images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

        #input GAN model

        #create saver to restore from checkpoints
        saver = tf.train.Saver()

        with tf.Session() as sess:
            #Restore the model from last checkpoints
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)

        #(re-)create export directory
        export_path=os.path.join(
            tf.compat.as_bytes(FLAGS.output_dir),
            tf.compat.as_bytes(str(FLAGS.model_version))
        )
        if os.path.exists(export_path):
            shutil.rmtree(export_path)

        #create model builder
        builder=tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)

        #create tensors info
        predict_tensor_inputs_info = tf.compat.v1.saved_model.utils.build_tensor_info(jpegs)
        predict_tensor_scores_info = tf.compat.v1.saved_model.utils.build_tensor_info(
            net.discriminator_out #prediction output넣기
        )

        #build prediction signature
        prediction_signature = (
            tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs={'images':predict_tensor_inputs_info},
                outputs={'scores':predict_tensor_scores_info},
                method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        #save the model
        legacy_init_op=tf.group(tf.compat.v1.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':prediction_signature
            },
            legacy_init_op=legacy_init_op
        )
        builder.save()

    print("Successfully exported GAN model version '{}' into '{}'".format(
        FLAGS.model_version, FLAGS.output_dir
    ))

if__name__ == '__main__':
    tf.compat.v1.app.run()

