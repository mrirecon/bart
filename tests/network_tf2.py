import bart_tf

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


def tf2_generate_resnet(path, model):

    import tensorflow as tf
    import numpy as np

    class ComplexConv3D(tf.Module):
        def __init__(self, filters, kernel_size, dummy_dim = False):
            super().__init__()
            # filters: 64, kernel_size: 3, stride: 1
            self.filters = filters
            self.kernel_size = kernel_size
            self.is_built = False
            self.dummy_dim = dummy_dim

        def __call__(self, input):

            if not(self.is_built):
                if self.dummy_dim:
                    shp = [1] + list(self.kernel_size) + [input.shape[-2], self.filters, 2]
                else:
                    shp = list(self.kernel_size) + [input.shape[-2], self.filters, 2]

                scale = np.sqrt(1 / (np.prod(self.kernel_size) * self.filters + input.shape[-2]))
                self.conv_weight = tf.Variable(tf.random.normal(shp, stddev=scale), name='w')
                self.is_built = True

            conv = self.conv_weight
            if self.dummy_dim:
                conv = tf.reshape(conv, conv.shape[1:])
            conv = real_from_complex_weights(conv)

            shp = list(input.shape[:-1])
            shp[-1] = shp[-1] * 2

            tmp = tf.reshape(input, shp)
            tmp = tf.nn.conv3d(tmp, conv, [1] * 5, "SAME")

            shp = list(tmp.shape)
            shp[-1] = shp[-1] // 2
            shp.append(2)

            return tf.reshape(tmp, shp)
    
    class ResBlock(tf.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ComplexConv3D(8, (1, 3, 3))
            self.conv2 = ComplexConv3D(8, (1, 3, 3), dummy_dim=True)
            self.conv3 = ComplexConv3D(1, (1, 3, 3))

        def __call__(self, input):

            shp = list(input.shape)
            shp.insert(-1, 1)

            out = tf.reshape(input, shp)
            out = self.conv1(out)
            out = tf.nn.relu(out)
            out = self.conv2(out)
            out = tf.nn.relu(out)
            out = self.conv3(out)
            out = input + tf.reshape(out, input.shape)
            return out

    bart_tf.tf2_export_module(ResBlock(), [32, 32, 1], path+"/"+model, trace_complex=False, batch_sizes=[2, 10])

tf2_generate_resnet("./", "tf2_resnet")


