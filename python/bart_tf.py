# Copyright 2022. Uecker Lab. University Center Göttingen.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# Moritz Blumenthal


def tf2_export_module(model, dims, path, trace_complex=True, batch_sizes = [10]):

    import tensorflow as tf
    import numpy as np

    class BartWrapper(tf.Module):
    
        def __init__(self, model, dims, batch_sizes = [10], vars_as_input = True, name=None):
        
            super(BartWrapper, self).__init__(name=name)

            self.model = model
            self.trace_complex = trace_complex
        
            self.dims_bart = [1] * 16
            self.dims_tf =   [1] * (len(dims) + 1)

            if not(trace_complex):

                self.dims_bart = self.dims_bart + [2]
                self.dims_tf = self.dims_tf + [2]
            
                for i in range(len(dims)):
                    self.dims_bart[len(self.dims_bart) - 2 - i] = dims[i]
                    self.dims_tf[len(self.dims_tf) - 2 - i] = dims[i]
                
                self.model(np.zeros(self.dims_tf, np.float32)) #run model ones to initialize weights

            else :
                for i in range(len(dims)):
                    self.dims_bart[len(self.dims_bart) - 2 - i] = dims[i]
                    self.dims_tf[len(self.dims_tf) - 2 - i] = dims[i]

                self.model(np.zeros(self.dims_tf, np.complex64)) #run model ones to initialize weights

            self.trace_complex = trace_complex
        
            if vars_as_input:
                self.vars = model.variables
            else:
                self.vars = []

            self.vars_rtoc = [] # variables for which a 0 imaginary part is stacked
            for var in self.vars:
                self.vars_rtoc.append(2 != var.shape[-1])
            
            self.sig = {}
            
            self.add_concrete_function()
            for bs in batch_sizes:
                self.add_concrete_function(bs)

        @tf.function
        def __call__(self, input, weights, grad_in):

            for i in range(len(weights)):
            
                wgh = weights[i]

                if (self.vars_rtoc)[i]:    
                    slc = [ slice(None, None, None) ]* len(wgh.shape)
                    slc[-1] = 0
                    wgh=wgh[tuple(slc)]
                
                self.model.variables[i].assign(wgh)


            with tf.GradientTape(persistent=True) as g:
                g.watch(input)

                shp = self.dims_tf.copy()
                shp[0] = input.shape[0]

                print("Tracing TensorFlow model with dims: {}".format(shp))
                out = self.model(tf.reshape(input, shp))
                out = tf.reshape(out, input.shape)

            result = {}
        
            result["output_0"] = out
            result["grad_0_0"] = g.gradient(out, input, grad_in)

            for i, input in enumerate(self.model.variables, 1):
                result["grad_{}_0".format(i)] = g.gradient(out, input, grad_in)
                
                if self.vars_rtoc[i - 1]:
                    tmp = result["grad_{}_0".format(i)]
                    result["grad_{}_0".format(i)] = tf.stack([tmp, tf.zeros_like(tmp)], axis = len(tmp.shape))
        
            return result


        def add_concrete_function(self, batch_size=1, name=None):

            dims = self.dims_bart.copy()
            dims[0] = batch_size

            if (self.trace_complex):
                signature_input = tf.TensorSpec(shape=dims, dtype=tf.complex64, name="input_0")
                signature_grad_ys = tf.TensorSpec(shape=dims, dtype=tf.complex64, name="grad_ys_0")
            else:
                signature_input = tf.TensorSpec(shape=dims, dtype=tf.float32, name="input_0")
                signature_grad_ys = tf.TensorSpec(shape=dims, dtype=tf.float32, name="grad_ys_0")

            signature_weight = []
            for i, var in enumerate(self.model.variables, 1):
                if self.vars_rtoc[i - 1]:
                    signature_weight.append(tf.TensorSpec(shape=list(var.shape)+[2], dtype=tf.float32, name="input_{}".format(i)))
                else:
                    signature_weight.append(tf.TensorSpec(shape=var.shape, dtype=tf.float32, name="input_{}".format(i)))

            if name is None:
                if 1 == batch_size:
                    name = "serving_default"
                else:
                    name = "serving_default_batch_{}".format(batch_size)

            self.sig[name] = self.__call__.get_concrete_function(signature_input, signature_weight, signature_grad_ys)


        def save_variables(self, path):
            
            weights = []
            for i, var in enumerate(self.variables):
                
                if (self.vars_rtoc[i]):
                    weights.append(var.numpy().astype(np.complex64)) 
                else:
                    weights.append(np.empty(var.shape[:-1], dtype=np.complex64))
                    slc = [ slice(None, None, None) ] * len(var.shape)
                    slc[-1] = 0
                    weights[-1].real = var.numpy()[tuple(slc)]
                    slc[-1] = 1
                    weights[-1].imag = var.numpy()[tuple(slc)]
                    
                    if 0 == len(weights[-1].shape):
                        weights[-1] = weights[-1].reshape([1]) 
            
                weights[-1] = np.transpose(weights[-1])
            
            if (0 < len(weights)):
                import cfl
                cfl.writemulticfl(path, weights)



        def save(self, path):

            tf.saved_model.save(self, path, signatures=self.sig)
            self.save_variables(path+"/bart_initial_weights")

            from tensorflow.python.tools import saved_model_utils
            meta_graph_def = saved_model_utils.get_meta_graph_def(path, "serve")

            with open(path + "/bart_config.dat", 'w') as f:

                for signature in list(self.sig):

                    inputs = meta_graph_def.signature_def[signature].inputs
                    outputs = meta_graph_def.signature_def[signature].outputs

                    f.write('# ArgumentNameMapping\n')
                    f.write('{}\n'.format(signature))

                    for bart_name in list(inputs):
                        f.write("{} {} {}\n".format(bart_name, inputs[bart_name].name.split(":")[0], inputs[bart_name].name.split(":")[1]))

                    for bart_name in list(outputs):
                        f.write("{} {} {}\n".format(bart_name, outputs[bart_name].name.split(":")[0], outputs[bart_name].name.split(":")[1]))

    
    BartWrapper(model, dims, batch_sizes).save(path)





def tf1_graph_attach_gradients(graph):

    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        import tensorflow as tf
        pass

    def op_exists(graph, name):  
        try:
            graph.get_operation_by_name(name)
            return True
        except KeyError:
            return False

    II = 0
    OO = 0

    inputs=[]
    outputs=[]
    grad_ys=[]

    while op_exists(graph, "input_"+str(II)):
        inputs.append(graph.get_tensor_by_name("input_{}:0".format(II)))
        II += 1
    
    while op_exists(graph, "output_"+str(OO)):
        outputs.append(graph.get_tensor_by_name("output_{}:0".format(OO)))
        if not(op_exists(graph, name='grad_ys_'+ str(OO))):
            with graph.as_default():
                grad_ys.append(tf.placeholder(outputs[-1].dtype, shape=outputs[-1].shape, name='grad_ys_'+ str(OO)))
        OO += 1

    print("{} Inputs found".format(II))
    print("{} Outputs found".format(OO))

    for i in range(II):
        for o in range(OO):
            if not(op_exists(graph, name='grad_{}_{}'.format(i, o))):
                with graph.as_default():
                    grad = tf.gradients(outputs[o], inputs[i], grad_ys[o])
                    tf.reshape(grad, inputs[i].shape, name='grad_{}_{}'.format(i, o))


def tf1_export_graph(graph, path, name, session=None):

    import os
    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        import tensorflow as tf
        pass

    
    tf1_graph_attach_gradients(graph)

    tf.train.write_graph(graph, path, name+'.pb', False)

    if session is not None:
        saver = tf.train.Saver()
        saver.save(session, os.path.join(path, name))


def tf1_convert_model(model_path, path, name):

    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        import tensorflow as tf
        pass

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_path) 

    tf1_graph_attach_gradients(sess.graph)
    tf1_export_graph(sess.graph, path, name, session=sess)