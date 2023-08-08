# Copyright 2022. Uecker Lab. University Center Göttingen.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# Moritz Blumenthal

import os
import numpy as np
import cfl

import tensorflow as tf2

try:
    import tensorflow.compat.v1 as tf1
except ImportError:
    import tensorflow as tf1
    pass

def tf2_export_module(model, dims, path, trace_complex=True):
    class BartWrapper(tf2.Module):
    
        def __init__(self, model, dims, vars_as_input = True, name=None):
        
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

            self.dims_tf[0] = -1
            self.dims_bart[0] = -1

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

        @tf2.function
        def __call__(self, input, weights, grad_in):

            for i in range(len(weights)):
            
                wgh = weights[i]

                if (self.vars_rtoc)[i]:    
                    slc = [ slice(None, None, None) ]* len(wgh.shape)
                    slc[-1] = 0
                    wgh=wgh[tuple(slc)]
                
                self.model.variables[i].assign(wgh)


            with tf2.GradientTape(persistent=True) as g:
                g.watch(input)

                print("Tracing TensorFlow model with dims: {}".format(input))

                res = tf2.reshape(input, self.dims_tf)

                outr = self.model(res)

                out = tf2.reshape(outr, self.dims_bart)

            result = {}
        
            result["output_0"] = out
            result["grad_0_0"] = g.gradient(out, input, grad_in)

            for i, input in enumerate(self.model.variables, 1):
                result["grad_{}_0".format(i)] = g.gradient(out, input, grad_in)
                
                if self.vars_rtoc[i - 1]:
                    tmp = result["grad_{}_0".format(i)]
                    result["grad_{}_0".format(i)] = tf2.stack([tmp, tf2.zeros_like(tmp)], axis = len(tmp.shape))
        
            return result


        def add_concrete_function(self, name=None):

            dims = self.dims_bart.copy()
            dims[0] = None

            if (self.trace_complex):
                signature_input = tf2.TensorSpec(shape=dims, dtype=tf2.complex64, name="input_0")
                signature_grad_ys = tf2.TensorSpec(shape=dims, dtype=tf2.complex64, name="grad_ys_0")
            else:
                signature_input = tf2.TensorSpec(shape=dims, dtype=tf2.float32, name="input_0")
                signature_grad_ys = tf2.TensorSpec(shape=dims, dtype=tf2.float32, name="grad_ys_0")

            signature_weight = []
            for i, var in enumerate(self.model.variables, 1):
                if self.vars_rtoc[i - 1]:
                    signature_weight.append(tf2.TensorSpec(shape=list(var.shape)+[2], dtype=tf2.float32, name="input_{}".format(i)))
                else:
                    signature_weight.append(tf2.TensorSpec(shape=var.shape, dtype=tf2.float32, name="input_{}".format(i)))

            if name is None:
                name = "serving_default"
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
                cfl.writemulticfl(path, weights)



        def save(self, path):

            tf2.saved_model.save(self, path, signatures=self.sig)
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

    
    BartWrapper(model, dims).save(path)

def tf2_add_signature(path, signature='serving_default'):

	from tensorflow.python.tools import saved_model_utils
	meta_graph_def = saved_model_utils.get_meta_graph_def(path, "serve")

	with open(path + "/bart_config.dat", 'w') as f:
		
		inputs = meta_graph_def.signature_def[signature].inputs
		outputs = meta_graph_def.signature_def[signature].outputs

		f.write('# ArgumentNameMapping\n')
		f.write('{}\n'.format(signature))

		for bart_name in list(inputs):
			f.write("{} {} {}\n".format(bart_name, inputs[bart_name].name.split(":")[0], inputs[bart_name].name.split(":")[1]))

		for bart_name in list(outputs):
			f.write("{} {} {}\n".format(bart_name, outputs[bart_name].name.split(":")[0], outputs[bart_name].name.split(":")[1]))

class TensorMap:
    def __init__(self, tensor, name, enforce_real = False):

        if isinstance(tensor, TensorMap):
            self.tensor = tensor.tensor
        else:
            self.tensor = tensor
        self.name = name

        if (self.tensor.shape[-1] != 2) and (self.tensor.dtype == tf1.float32):
            self.type = "REAL"
        else:
            self.type = "COMPLEX"

        if isinstance(tensor, TensorMap):
            self.type = tensor.type

        if enforce_real:
            self.type = "REAL"

    def export(self):
        n = self.tensor.name
        return "{} {} {} {}".format(self.name, n.split(":")[0], n.split(":")[1], self.type)

def tf1_export_tensor_mapping(path, name, mapping, signature="serving_default"):
    with open(path + "/" + name + ".map", 'w') as f:
        f.write('# ArgumentNameMapping\n')
        f.write('{}\n'.format(signature))
        for map in mapping:
            f.write('{}\n'.format(map.export()))

def tf1_op_exists(graph, name):
    try:
        graph.get_operation_by_name(name)
        return True
    except KeyError:
        return False

def tf1_find_tensors(graph, inputs, outputs):
    
    if inputs is None:
        II = 0
        inputs = []
        while tf1_op_exists(graph, "input_"+str(II)):
            inputs.append(graph.get_tensor_by_name("input_{}:0".format(II)))
            II += 1

    if outputs is None:
        OO = 0
        outputs = []
        while tf1_op_exists(graph, "output_"+str(OO)):
            outputs.append(graph.get_tensor_by_name("output_{}:0".format(OO)))
            OO += 1

    for i in range(len(inputs)):
        inputs[i] = TensorMap(inputs[i], "input_"+str(i))

    for i in range(len(outputs)):
        outputs[i] = TensorMap(outputs[i], "output_"+str(i))

    return inputs, outputs


def tf1_graph_attach_gradients(graph, inputs, outputs):

    grad_tensors=[]

    for o, out in enumerate(outputs):
        with graph.as_default():
            gy = tf1.placeholder(out.tensor.dtype, shape=out.tensor.shape, name='grad_ys_'+ str(o))
            grad_tensors.append(TensorMap(gy, 'grad_ys_'+ str(o), out.type == "REAL"))

    for i, inp in enumerate(inputs):
        for o, out in enumerate(outputs):
            name = 'grad_{}_{}'.format(i, o)
            with graph.as_default():
                grad = tf1.gradients(out.tensor, inp.tensor, grad_tensors[o].tensor)
                grad = tf1.reshape(grad, tf1.shape(inp.tensor), name='grad_{}_{}'.format(i, o))
                grad_tensors.append(TensorMap(grad, name, inp.type == "REAL"))
    
    return grad_tensors


def tf1_export_graph(path, graph = None, session=None, inputs=None, outputs=None, name=None, attach_gradients=True):

    if graph is None:
        graph = tf1.get_default_graph()

    if name is None:
        name = os.path.basename(os.path.normpath(path))

    inputs, outputs = tf1_find_tensors(graph, inputs, outputs)

    mappings = []
    if attach_gradients:
        mappings = tf1_graph_attach_gradients(graph, inputs, outputs)

    mappings += inputs
    mappings += outputs

    tf1.train.write_graph(graph, path, name+'.pb', False)

    if session is not None:
        saver = tf1.train.Saver()
        saver.save(session, os.path.join(path, name))
    else:
        if (tf1_op_exists(graph, "save/restore_all")):
            print("WARNING: No weights are stored with the graph!\nWARNING: BART probably will not be able to load the graph.")

    tf1_export_tensor_mapping(path, name, mappings)


def tf1_convert_model(model_path, path, name):

    sess = tf1.Session()
    saver = tf1.train.Saver()
    saver.restore(sess, model_path) 

    tf1_graph_attach_gradients(sess.graph)
    tf1_export_graph(sess.graph, path, name, session=sess)
