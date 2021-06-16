/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <stdio.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"

#include "num/multind.h"

#include "nlops/nlop.h"

#ifdef _WIN32
#include <stdint.h>
#endif

#ifdef TENSORFLOW
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_tensor.h"
#endif

#include "tf_wrapper.h"

#ifndef FL_SIZE
#define FL_SIZE sizeof(float)
#define CFL_SIZE sizeof(complex float)
#endif

//#define TF_AUTOGRAD 1
// The TensorFlow C API does not support all gradients
// Thus we require that the TensorFlow graph is annotated with gradients


#ifdef TENSORFLOW
static int product(int n, const int64_t ar[n])
{
    int64_t result = 1;

    for (int i = 0; i < n; i++)
	result = result * ar[i];

    return result;
}
#endif

// function to read network/graph definition from binary protobuf file

#ifdef TENSORFLOW
static void free_buf(void* data, size_t size)
{
	unmap_raw(data, size);
}

static TF_Graph* load_graph(const char* name, TF_Status* status)
{
	TF_Buffer* buf = TF_NewBuffer();

	buf->data = private_raw(&buf->length, name);
	buf->data_deallocator = free_buf;

	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

	TF_Graph* graph = TF_NewGraph();
	TF_GraphImportGraphDef(graph, buf, opts, status);

	TF_DeleteBuffer(buf);
	TF_DeleteImportGraphDefOptions(opts);

	if (TF_GetCode(status) != TF_OK)
		error("Loading TensorFlow graph failed: %s\n", TF_Message(status));

	debug_printf(DP_DEBUG1, "TensorFlow graph loaded from file %s.\n", name);

	return graph;
}


static TF_Session* create_session(TF_Graph* graph, TF_Status* status)
{
	TF_SessionOptions* opt = TF_NewSessionOptions();
#if 0
	TF_SetConfig(opt, config, clen, status);

	if (TF_GetCode(status) != TF_OK)
		debug_printf(DP_INFO, "GPU selection failed\n");
#endif
	TF_Session* sess = TF_NewSession(graph, opt, status);

	TF_DeleteSessionOptions(opt);

	if (TF_GetCode(status) != TF_OK)
		error("Unable to create TensorFlow session %s\n", TF_Message(status));

	debug_printf(DP_DEBUG1, "TensorFlow session created.\n");

	return sess;
}

#if 0
static void deallocator(void* ptr, size_t len, void* arg)
{
	xfree(ptr);
	UNUSED(len); UNUSED(arg);
}
#else
static void deallocator(void* ptr, size_t len, void* arg)
{
	TF_TString_Dealloc(ptr);
	UNUSED(len); UNUSED(arg);
}
#endif

// function to restore trained weights
static void restore_session(TF_Graph* graph, TF_Status *status, TF_Session *sess, const char* ckpt_path)
{
	TF_Operation* checkpoint_op = TF_GraphOperationByName(graph, "save/Const");

	const TF_Operation* restore_op = TF_GraphOperationByName(graph, "save/restore_all");

#if 0
	size_t checkpoint_path_str_len = strlen(ckpt_path);
	size_t encoded_size = TF_StringEncodedSize(checkpoint_path_str_len);
	size_t total_size = sizeof(int64_t) + encoded_size;

	char* input_encoded = xmalloc(total_size);

	memset(input_encoded, 0, total_size);

	TF_StringEncode(ckpt_path, checkpoint_path_str_len, input_encoded + sizeof(int64_t), encoded_size, status);

	if (TF_GetCode(status) != TF_OK)
		error("Something wrong with encoding: %s", TF_Message(status));

	TF_Tensor* path_tensor = TF_NewTensor(TF_STRING, NULL, 0, input_encoded, total_size, &deallocator, 0);
#else
	TF_TString path_string;
	TF_TString_Init(&path_string);
	TF_TString_Copy(&path_string, ckpt_path, strlen(ckpt_path));
	TF_Tensor* path_tensor = TF_NewTensor(TF_STRING, NULL, 0, &path_string, TF_TString_GetSize(&path_string), &deallocator, 0);
#endif
	TF_Output run_path;
	run_path.oper = checkpoint_op;
	run_path.index = 0;

	TF_SessionRun(sess,	/* RunOptions */ NULL,
				/* Input tensors */ &run_path, &path_tensor, 1,
				/* Output tensors */ NULL, NULL, 0,
				/* Target operations */ &restore_op, 1,
				/* RunMetadata */ NULL,
				/* Output status */ status);

	TF_DeleteTensor(path_tensor);

	if (TF_GetCode(status) != TF_OK)
		error("Unable to run restore TensorFlow session: %s\n", TF_Message(status));

	debug_printf(DP_DEBUG1, "TensorFlow session restored from path %s.\n", ckpt_path);
}
#endif



#ifndef TENSORFLOW
struct TF_Output { int dummy; };
typedef int TF_Graph;
typedef int TF_Status;
typedef int TF_Session;
typedef int TF_Tensor;
#endif


static TF_Tensor* tensor_allocate(int N, const long dims[N])
{
	long dims2[N];
	assert(0 < N);

	for (int i = 0; i < N; i++)
		dims2[i] = dims[N - i - 1];

	assert(1 == dims2[N - 1]);
	dims2[N - 1] = 2;

#ifdef TENSORFLOW
	size_t size = product(N, dims2) * FL_SIZE;

	return TF_AllocateTensor(TF_FLOAT, dims2, N, size);
#else
	return NULL;
#endif
}
struct tf_s {

	INTERFACE(nlop_data_t);

	int nr_inputs;
	int nr_outputs;

#ifdef TENSORFLOW
	TF_Session* sess;
	TF_Status* status;
	TF_Graph* graph;
#endif

	TF_Tensor* const* input_tensors;

	struct TF_Output *inputs_op;
	struct TF_Output *outputs_op;
	struct TF_Output *grad_op;
	struct TF_Output *grad_ys_op;

	int *nr_out_dim;
	int *nr_in_dim;

	const int64_t **out_dims_tf;
	const int64_t **in_dims_tf;
};

DEF_TYPEID(tf_s);

static void tf_forward(const nlop_data_t* _data, int N, complex float* args[N])
{
	auto data = CAST_DOWN(tf_s, _data);

	assert(data->nr_inputs + data->nr_outputs == N);
#ifdef TENSORFLOW
	TF_Tensor* output_tensors[data->nr_outputs];

	for (int i = 0; i < data->nr_inputs; i++)
		md_copy(data->nr_in_dim[i], data->in_dims_tf[i], TF_TensorData(data->input_tensors[i]), args[i + data->nr_outputs], CFL_SIZE);

	TF_SessionRun(data->sess,
				/* RunOptions */ NULL,
				/* Input tensors */ data->inputs_op, data->input_tensors, data->nr_inputs,
				/* Output tensors */ data->outputs_op, output_tensors, data->nr_outputs,
				/* Target operations */ NULL, 0,
				/* RunMetadata */ NULL,
				/* Output status */ data->status);

	if (TF_GetCode(data->status) != TF_OK)
		error("Running TensorFlow failed: %s\n", TF_Message(data->status));

	for (int i = 0; i < data->nr_outputs; i++) {

		md_copy(data->nr_out_dim[i], data->out_dims_tf[i], args[i], TF_TensorData(output_tensors[i]), CFL_SIZE);

		TF_DeleteTensor(output_tensors[i]);
	}
#else
	UNUSED(N); UNUSED(args);

	error("TensorFlow support not available.\n");
#endif
}

static void tf_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(tf_s, _data);

	error("Calling the derivative of a TensorFlow graph is not supported.");

	UNUSED(data);
	UNUSED(dst);
	UNUSED(src);
	UNUSED(o);
	UNUSED(i);
}

static void tf_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(tf_s, _data);

#ifdef TENSORFLOW
	struct TF_Output inp_ops[data->nr_inputs + 1];
	struct TF_Tensor* inp_tensors[data->nr_inputs + 1];

	for (int j = 0; j < data->nr_inputs; j++) {

		inp_ops[j] = data->inputs_op[j];
		inp_tensors[j] = data->input_tensors[j];
	}

	inp_ops[data->nr_inputs] = data->grad_ys_op[o];
	inp_tensors[data->nr_inputs] = tensor_allocate(data->nr_out_dim[o], data->out_dims_tf[o]);

	md_copy(data->nr_out_dim[o], data->out_dims_tf[o], TF_TensorData(inp_tensors[data->nr_inputs]), src, CFL_SIZE);

	struct TF_Tensor* out_tensor[1];

	TF_SessionRun(data->sess,
				/* RunOptions */ NULL,
				/* Input tensors */ inp_ops, inp_tensors, data->nr_inputs + 1,
				/* Output tensors */ &(data->grad_op[i]), out_tensor, 1,
				/* Target operations */ NULL, 0,
				/* RunMetadata */ NULL,
				/* Output status */ data->status);

	if (TF_GetCode(data->status) != TF_OK)
		error("Running TensorFlow failed: %s\n", TF_Message(data->status));

	TF_DeleteTensor(inp_tensors[data->nr_inputs]);

	md_copy(data->nr_in_dim[i], data->in_dims_tf[i], dst, TF_TensorData(out_tensor[0]), CFL_SIZE);

	TF_DeleteTensor(out_tensor[0]);
#else
	UNUSED(data); UNUSED(o); UNUSED(i); UNUSED(dst); UNUSED(src);

	error("TensorFlow support not available.\n");
#endif
}


static void tf_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(tf_s, _data);

#ifdef TENSORFLOW
	for (int i = 0; i < data->nr_inputs; i++)
		TF_DeleteTensor(data->input_tensors[i]);

	TF_DeleteGraph(data->graph);
	TF_DeleteSession(data->sess, data->status);
#endif
	xfree(data->input_tensors);

	xfree(data->inputs_op);
	xfree(data->outputs_op);
	xfree(data->grad_op);
	xfree(data->grad_ys_op);

	xfree(data->nr_out_dim);
	xfree(data->nr_in_dim);

	for (int i = 0; i < data->nr_inputs; i++)
		xfree(data->in_dims_tf[i]);

	xfree(data->in_dims_tf);

	for (int i = 0; i < data->nr_outputs; i++)
		xfree(data->out_dims_tf[i]);

	xfree(data->out_dims_tf);

	xfree(data);
};



struct tf_arg {

	struct TF_Output out;
	int N;
	const int64_t* dims;
};



static struct tf_arg process_arg(TF_Graph* graph, const char* name, TF_Status* status)
{
#ifdef TENSORFLOW
	struct tf_arg arg;

	arg.out = (struct TF_Output){ TF_GraphOperationByName(graph, name), 0 };

	if (NULL == arg.out.oper)
		error("Graph operation %s missing.\n", name);

	arg.N = TF_GraphGetTensorNumDims(graph, arg.out, status);

	if (TF_GetCode(status) != TF_OK)
		error("Getting TensorFlow dimensions failed: %s\n", TF_Message(status));
#if 0
	if (0 == arg.N)
		error("Graph operaton %s missing or incorrect.\n", name);
#endif
	long tdims[arg.N ?: 1];

	TF_GraphGetTensorShape(graph, arg.out, tdims, arg.N, status);

	if (TF_GetCode(status) != TF_OK)
		error("Getting TensorFlow shape failed: %s\n", TF_Message(status));

	if (0 == arg.N) {	// create a scalar

		error("TensorFlow: Real scalar arguments are not supported! Stack with zero_like to construct complex argument!");
		arg.N = 1;
		tdims[0] = 2;
	}

	PTR_ALLOC(int64_t[arg.N], dims);

	for (int i = 0; i < arg.N; i++) // convert to Fortran order
		(*dims)[i] = tdims[arg.N - i - 1];

	if (2 != (*dims)[0])
		error("TensorFlow: Last dimension must have size 2 for real and imaginary part!\nStack with zero_like to construct complex argument!");

	(*dims)[0] = 1;


	arg.dims = *PTR_PASS(dims);
#else
	struct tf_arg arg = { 0 };

	UNUSED(graph); UNUSED(name); UNUSED(status);
#endif

	return arg;
}

static bool cmp_arg(struct tf_arg arg1, struct tf_arg arg2)
{

	bool result = true;

	for (int i = 0; i < MIN(arg1.N, arg2.N); i++)
		result = result && (arg1.dims[i] == arg2.dims[i]);

	for (int i = MIN(arg1.N, arg2.N); i < arg1.N; i++)
		result = result && (1 == arg1.dims[i]);

	for (int i = MIN(arg1.N, arg2.N); i < arg2.N; i++)
		result = result && (1 == arg2.dims[i]);

	return result;
}

#ifdef TF_AUTOGRAD

static void tf_add_placeholder_same_shape(TF_Graph* graph, const char* name, TF_Status* status, struct TF_Output out)
{
#ifdef TENSORFLOW
	TF_OperationDescription* desc = TF_NewOperation(graph, "Placeholder", name);

	int N = TF_GraphGetTensorNumDims(graph, out, status);

	if (TF_GetCode(status) != TF_OK)
		error("Add Tensorflow Placeholder failed: %s\n", TF_Message(status));

	long tdims[N ?: 1];
	TF_GraphGetTensorShape(graph, out, tdims, N, status);

	if (TF_GetCode(status) != TF_OK)
		error("Add Tensorflow Placeholder failed: %s\n", TF_Message(status));

	TF_DataType type = TF_OperationOutputType(out);

	TF_SetAttrType(desc, "dtype", type);
	TF_SetAttrShape(desc, "shape", tdims, N);
	TF_FinishOperation(desc, status);

	if (TF_GetCode(status) != TF_OK)
		error("Add Tensorflow Placeholder failed: %s\n", TF_Message(status));
#else
	UNUSED(graph);
	UNUSED(name);
	UNUSED(status);
	UNUSED(out);
#endif
}
#endif

const struct nlop_s* nlop_tf_create(int OO, int II, const char* path, bool session)
{
	assert(1 == OO);

	// load graph, restore session

	int plen = strlen(path) + 4;

	char graph_path[plen];
	int rlen = snprintf(graph_path, plen, "%s.pb", path);
	assert(rlen < plen);
#ifdef TENSORFLOW
	TF_Status* status = TF_NewStatus();

	TF_Graph* graph = load_graph(graph_path, status);

	TF_Session* sess = create_session(graph, status);

	if (session)
		restore_session(graph, status, sess, path);
#else
	UNUSED(session);
	TF_Status* status;
	TF_Graph* graph;
#endif

	/*** handle outputs and grad_ys ***/

	// outputs
	int ON = 1;
	int ON_arr[OO];

	PTR_ALLOC(struct TF_Output[OO], outputs_op);
	PTR_ALLOC(int[OO], nr_out_dim);
	PTR_ALLOC(const int64_t*[OO], out_dims_tf);
	PTR_ALLOC(struct TF_Output[OO], grad_ys_op);

	for (int i = 0; i < OO; i++) {

		char out_name[20];
		sprintf(out_name, "output_%d", i);
		struct tf_arg arg = process_arg(graph, out_name, status);

		ON_arr[i] = arg.N;
		ON = MAX(ON, ON_arr[i]);

		(*outputs_op)[i] = arg.out;
		(*nr_out_dim)[i] = arg.N;
		(*out_dims_tf)[i] = arg.dims;

#ifdef TF_AUTOGRAD
		char grad_ys_name[20];
		sprintf(grad_ys_name, "grad_ys_bart_%d", i);

		tf_add_placeholder_same_shape(graph, grad_ys_name, status, arg.out);
#else
		char grad_ys_name[20];
		sprintf(grad_ys_name, "grad_ys_%d", i);
#endif

		struct tf_arg arg_grad_y = process_arg(graph, grad_ys_name, status);

		if (!cmp_arg(arg, arg_grad_y) || (arg.N != arg_grad_y.N))
			error("Tensorflow output and corresponding gradient input do not have the same shape!");

		(*grad_ys_op)[i] = arg_grad_y.out;
	}

	PTR_ALLOC(struct tf_s, data);
	SET_TYPEID(tf_s, data);
#ifdef TENSORFLOW
	data->sess = sess;
	data->status = status;
	data->graph = graph;
#endif
	data->nr_inputs = II;
	data->nr_outputs = OO;

	data->outputs_op = *PTR_PASS(outputs_op);
	data->nr_out_dim = *PTR_PASS(nr_out_dim);
	data->out_dims_tf = *PTR_PASS(out_dims_tf);
	data->grad_ys_op = *PTR_PASS(grad_ys_op);

	// handle inputs and grad
	int IN = 1;
	int IN_arr[II];

	PTR_ALLOC(struct TF_Output[II], inputs_op);
	PTR_ALLOC(TF_Tensor*[II], input_tensors);
	PTR_ALLOC(int[II], nr_in_dim);
	PTR_ALLOC(const int64_t *[II], in_dims_tf);
	PTR_ALLOC(struct TF_Output[II], grad_op);

	for (int i = 0; i < II; i++) {

		char in_name[20];
		sprintf(in_name, "input_%d", i);

		struct tf_arg arg = process_arg(graph, in_name, status);

		IN_arr[i] = arg.N;
		IN = MAX(IN, IN_arr[i]);

		(*input_tensors)[i] = tensor_allocate(arg.N, arg.dims);
		(*inputs_op)[i] = arg.out;
		(*nr_in_dim)[i] = arg.N;
		(*in_dims_tf)[i] = arg.dims;

#ifdef TF_AUTOGRAD
#ifdef TENSORFLOW
		TF_AddGradients(graph, data->outputs_op, 1, &(*inputs_op)[i], 1, data->grad_ys_op, data->status, &(*grad_op)[i]);

		if (TF_OK != TF_GetCode(status))
			error("Add Tensorflow Gradient failed: %s\n", TF_Message(status));
#endif
#else

		char grad_name[20];
		sprintf(grad_name, "grad_%d", i);

		struct tf_arg arg_grad = process_arg(graph, grad_name, status);

		if (!cmp_arg(arg, arg_grad))
			error("Tensorflow input and corresponding gradient do not have the same shape!");

		(*grad_op)[i] = arg_grad.out;
#endif
	}

	data->inputs_op = *PTR_PASS(inputs_op);
	data->input_tensors = *PTR_PASS(input_tensors);
	data->nr_in_dim = *PTR_PASS(nr_in_dim);
	data->in_dims_tf = *PTR_PASS(in_dims_tf);
	data->grad_op = *PTR_PASS(grad_op);



	long nl_odims[OO][ON];
	long nl_idims[II][IN];

	for (int i = 0; i < OO; i++)
		for (int j = 0; j < ON; j++)
			nl_odims[i][j] = (j < ON_arr[i]) ? data->out_dims_tf[i][j] : 1;

	for (int i = 0; i < II; i++)
		for (int j = 0; j < IN; j++)
			nl_idims[i][j] = (j < IN_arr[i]) ? data->in_dims_tf[i][j] : 1;


	nlop_der_fun_t deriv[II][OO];
	nlop_der_fun_t adjoint[II][OO];
	nlop_der_fun_t normal[II][OO];
	nlop_p_fun_t norm_inv[II][OO];

	for (int i = 0; i < II; i++) {
		for (int o = 0; o < OO; o++) {

			deriv[i][o] = tf_der;
			adjoint[i][o] = tf_adj;
			normal[i][o] = NULL;
			norm_inv[i][o] = NULL;
		}
	}

	const struct nlop_s* result = nlop_generic_create(	OO, ON, nl_odims, II, IN, nl_idims,
								CAST_UP(PTR_PASS(data)), tf_forward, deriv, adjoint, normal, norm_inv, tf_del);
	return result;
}




