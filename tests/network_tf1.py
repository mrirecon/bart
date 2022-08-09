import bart_tf

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def real_from_complex_weights(wgh):
    import tensorflow as tf

    shp = wgh.shape

    filter_depth, filter_height, filter_width, in_channels, out_channels, tmp = shp
    size = [filter_depth, filter_height, filter_width, in_channels, out_channels, 1]
    
    rwgh=tf.slice(wgh, begin=[0,0,0,0,0,0], size = size)
    iwgh=tf.slice(wgh, begin=[0,0,0,0,0,1], size = size)

    rwgh = tf.reshape(rwgh, [filter_depth, filter_height, filter_width, in_channels, 1, out_channels, 1])
    iwgh = tf.reshape(iwgh, [filter_depth, filter_height, filter_width, in_channels, 1, out_channels, 1])

    wgh = tf.concat([tf.concat([rwgh, iwgh], 6), tf.concat([-iwgh, rwgh], 6)], 4)

    return tf.reshape(wgh, [filter_depth, filter_height, filter_width, 2 * in_channels, 2 * out_channels])


def tf1_generate_resnet(path, model):

    tf.reset_default_graph()

    batch_size = 2
    image_shape = [1, 32, 32]
    img_channel = 1

    img = tf.placeholder(tf.float32, shape=[batch_size] + [1] * 12 + image_shape+[2], name='input_0')
    img_t = tf.reshape(img, [batch_size] + image_shape + [img_channel * 2])

    conv_0 = tf.placeholder(tf.float32, shape=[1, 3, 3, 1, 8, 2], name='input_1')
    conv_i = tf.placeholder(tf.float32, shape=[1, 1, 3, 3, 8, 8, 2], name='input_2')
    conv_n = tf.placeholder(tf.float32, shape=[1, 3, 3, 8, 1, 2], name='input_3')

    conv_is = tf.reshape(conv_i, [1, 3, 3, 8, 8, 2])

    wghts = []
    wghts.append(real_from_complex_weights(conv_0))
    wghts.append(real_from_complex_weights(conv_is))
    wghts.append(real_from_complex_weights(conv_n))

    out = img_t
    for wgh in wghts:
        out = tf.nn.conv3d(out, wgh, [1] * 5, "SAME")
        if wgh != wghts[-1]:
            out = tf.nn.relu(out)
        
    out = out + img_t
    out = tf.reshape(out, img.shape, name='output_0')

    bart_tf.tf1_export_graph(tf.get_default_graph(), path, model)

tf1_generate_resnet("./", "tf1_resnet")



