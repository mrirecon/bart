/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/shrdptr.h"
#include "misc/list.h"

#include "num/flpmath.h"
#include "num/multind.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#ifdef _OPENMP
#include "omp.h"
#endif

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

#ifndef TENSORFLOW

const struct tf_shared_graph_s* tf_shared_graph_create(const char* path, const char* signature_key, bool session)
{
	UNUSED(path); UNUSED(signature_key); UNUSED(session);
	error("BART is build without TensorFlow support!\nRebuild with \"TENSORFLOW=1\"\n");
}

void tf_shared_graph_free(const struct tf_shared_graph_s* x)
{
	UNUSED(x);
	error("BART is build without TensorFlow support!\nRebuild with \"TENSORFLOW=1\"\n");
}

const char* tf_shared_graph_get_init_path(const struct tf_shared_graph_s* x)
{
	UNUSED(x);
	error("BART is build without TensorFlow support!\nRebuild with \"TENSORFLOW=1\"\n");
}


const struct nlop_s* nlop_tf_shared_create(const struct tf_shared_graph_s* graph)
{
	UNUSED(graph);
	error("BART is build without TensorFlow support!\nRebuild with \"TENSORFLOW=1\"\n");
}

const struct nlop_s* nlop_tf_create(int nr_outputs, int nr_inputs, const char* path, bool session)
{
	UNUSED(nr_inputs); UNUSED(nr_outputs); UNUSED(path); UNUSED(session);
	error("BART is build without TensorFlow support!\nRebuild with \"TENSORFLOW=1\"\n");
}

#else 


static int product(int n, const int64_t ar[n])
{
    int64_t result = 1;

    for (int i = 0; i < n; i++)
	result = result * ar[i];

    return result;
}


// function to read network/graph definition from binary protobuf file
/*
Python code to generate session config for selecting GPUs (https://github.com/tensorflow/tensorflow/issues/13853):

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 0})
config.gpu_options.allow_growth=True
config.intra_op_parallelism_threads = 9
config.inter_op_parallelism_threads = 9
result = list(map(hex, config.SerializeToString()))
print("uint8_t no_gpu[] = { "+ str(len(result))+", "+ ", ".join(result)+" };")

for i in range(16):
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    config.gpu_options.visible_device_list=str(i)
	config.intra_op_parallelism_threads = 9
	config.inter_op_parallelism_threads = 9
    result = list(map(hex, config.SerializeToString()))
    print('uint8_t gpu_{}[] = {{ '.format(i)+ str(len(result))+", "+ ", ".join(result)+" };")

Afterwards replace 0x9 with threads.
This seems to work upt to threads 127
*/

static TF_SessionOptions* get_session_opts(void)
{
	int threads = 1;

#ifdef _OPENMP
	threads = omp_get_max_threads();
	threads = MIN(127, threads);
#endif

	uint8_t no_gpu[] = { 19, 0xa, 0x7, 0xa, 0x3, 0x47, 0x50, 0x55, 0x10, 0x0, 0x10, threads, 0x28, threads, 0x32, 0x2, 0x20, 0x1, 0x38, 0x1 };
	uint8_t gpu_0[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x30, 0x38, 0x1 };
	uint8_t gpu_1[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x31, 0x38, 0x1 };
	uint8_t gpu_2[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x32, 0x38, 0x1 };
	uint8_t gpu_3[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x33, 0x38, 0x1 };
	uint8_t gpu_4[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x34, 0x38, 0x1 };
	uint8_t gpu_5[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x35, 0x38, 0x1 };
	uint8_t gpu_6[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x36, 0x38, 0x1 };
	uint8_t gpu_7[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x37, 0x38, 0x1 };
	uint8_t gpu_8[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x38, 0x38, 0x1 };
	uint8_t gpu_9[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x39, 0x38, 0x1 };
	uint8_t gpu_10[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x30, 0x38, 0x1 };
	uint8_t gpu_11[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x31, 0x38, 0x1 };
	uint8_t gpu_12[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x32, 0x38, 0x1 };
	uint8_t gpu_13[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x33, 0x38, 0x1 };
	uint8_t gpu_14[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x34, 0x38, 0x1 };
	uint8_t gpu_15[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x35, 0x38, 0x1 };
	uint8_t* gpu[] = { gpu_0, gpu_1, gpu_2, gpu_3, gpu_4, gpu_5, gpu_6, gpu_7, gpu_8, gpu_9, gpu_10, gpu_11, gpu_12, gpu_13, gpu_14, gpu_15 };
	
	uint8_t* config = no_gpu;

#ifdef USE_CUDA
	if (1 == cuda_num_devices())
		config = gpu[cuda_get_device_internal_unchecked()];
	
	if (1 < cuda_num_devices())
		error("TensorFlow Wrapper does not support multiple GPUs!\n");
#else
	UNUSED(gpu);
#endif

	TF_Status* status = TF_NewStatus();
	TF_SessionOptions* sess_opts = TF_NewSessionOptions();
		
	TF_SetConfig(sess_opts, (void*)(config + 1), *config, status);
	
	if (TF_GetCode(status) != TF_OK)
		error("Unable to parse session option config: \n", TF_Message(status));
	
	TF_DeleteStatus(status);
	
	return sess_opts;
}

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
	TF_SessionOptions* opt = get_session_opts();

	TF_Session* sess = TF_NewSession(graph, opt, status);

	TF_DeleteSessionOptions(opt);

	if (TF_GetCode(status) != TF_OK)
		error("Unable to create TensorFlow session %s\n", TF_Message(status));

	debug_printf(DP_DEBUG1, "TensorFlow session created.\n");

	return sess;
}


static void deallocator(void* ptr, size_t len, void* arg)
{
	TF_TString_Dealloc(ptr);
	UNUSED(len); UNUSED(arg);
}


// function to restore trained weights
static void restore_session(TF_Graph* graph, TF_Status *status, TF_Session *sess, const char* ckpt_path)
{
	TF_Operation* checkpoint_op = TF_GraphOperationByName(graph, "save/Const");

	const TF_Operation* restore_op = TF_GraphOperationByName(graph, "save/restore_all");


	TF_TString path_string;
	TF_TString_Init(&path_string);
	TF_TString_Copy(&path_string, ckpt_path, strlen(ckpt_path));
	TF_Tensor* path_tensor = TF_NewTensor(TF_STRING, NULL, 0, &path_string, TF_TString_GetSize(&path_string), &deallocator, 0);

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

struct tf_arg_map_s {

	const char* bart_name;
	const char* tf_name;
	int tf_index;
};

static struct tf_arg_map_s* tf_arg_map_create(const char* bname, const char* tname, int index)
{
	PTR_ALLOC(struct tf_arg_map_s, tm);

	tm->bart_name = ptr_printf("%s", bname);
	tm->tf_name = ptr_printf("%s", tname);
	tm->tf_index = index;

	return PTR_PASS(tm);
}

static void tf_arg_map_free(struct tf_arg_map_s* tm)
{
	xfree(tm->bart_name);
	xfree(tm->tf_name);

	xfree(tm);
}


static list_t read_name_mapping(const char * filename, const char* signature_key)
{
	if (NULL == filename)
		return NULL;

	int fd;
	if (-1 == (fd = open(filename, O_RDONLY)))
		error("TensorFlow config file %s not found!\n", filename);
	
	char config[4097];
	memset(config, 0, 4097);

	int max;
	if (0 > (max = read(fd, config, 4096)))
		error("TensorFlow config file %s too large!\n", filename);

	int pos = 0;
	int delta = 0;

	list_t arg_map = list_create();

	while (true) {

		// skip lines not starting with '#'

		while ('#' != config[pos]) {

			if ('\0' == config[pos])
				goto out;

			if (0 != sscanf(config + pos, "%*[^\n]\n%n", &delta))
				error("Could not parse TensorFlow config file for BART!\n");

			if (0 == delta)
				goto out;

			pos += delta;
		}

		char keyword[32];

		if (1 == sscanf(config + pos, "# %31s\n%n", keyword, &delta)) {

			pos += delta;

			if (0 == strcmp(keyword, "ArgumentNameMapping")) {

				char signature[80];

				if ((1 == sscanf(config + pos, "%79s\n%n", signature, &delta)) && (0 == strcmp(signature_key, signature))) {

					debug_printf(DP_INFO, "Found signature \"%s\" in config.\n", signature);

					pos += delta;
					char bart_name[80];
					char tf_name[80];
					int index;

					while (3 == sscanf(config + pos, "%79s %79s %d\n%n", bart_name, tf_name, &index, &delta)) {
 
 						debug_printf(DP_DEBUG1, "TensorFlow input mapping: %s %s %d\n", bart_name, tf_name, index);
						list_append(arg_map, tf_arg_map_create(bart_name, tf_name, index));

						pos += delta;
					}

					if(3 == sscanf(config + pos, "%79s %79s:%d%n", bart_name, tf_name, &index, &delta)) {

						debug_printf(DP_DEBUG1, "TensorFlow output mapping: %s %s %d\n", bart_name, tf_name, index);
						list_append(arg_map, tf_arg_map_create(bart_name, tf_name, index));

						pos += delta;
					}
				}
			}

		} else {

			// skip this line

			if (0 != sscanf(config + pos, "%*[^\n]\n%n", &delta))
				error("Could not parse TensorFlow config file for BART!\n");

			if (0 == delta)
				goto out;

			pos += delta;
		}
	}

out:
	return arg_map;
}

struct tf_shared_graph_s {

	struct shared_obj_s sptr;

	TF_Status* status;
	TF_Graph* graph;
	TF_Session* sess;

	list_t arg_name_map;

	const char* weight_init;
};

static void tf_shared_graph_del(const struct shared_obj_s* sptr)
{
	const struct tf_shared_graph_s* x = CONTAINER_OF(sptr, const struct tf_shared_graph_s, sptr);

	TF_DeleteGraph(x->graph);
	TF_DeleteSession(x->sess, x->status);
	TF_DeleteStatus(x->status);
	
	if (NULL != x->arg_name_map) {

		while (0 < list_count(x->arg_name_map))
			tf_arg_map_free(list_pop(x->arg_name_map));
	
		list_free(x->arg_name_map);
	}
	

	xfree(x);
}

static const struct tf_shared_graph_s* tf_shared_graph_ref(const struct tf_shared_graph_s* x)
{
	if (NULL != x)
		shared_obj_ref(&x->sptr);

	return x;
}

void tf_shared_graph_free(const struct tf_shared_graph_s* x)
{
	if (NULL == x)
		return;

	shared_obj_destroy(&x->sptr);
}

const char* tf_shared_graph_get_init_path(const struct tf_shared_graph_s* x)
{
	if (NULL == x)
		return NULL;

	return x->weight_init;
}

const struct tf_shared_graph_s* tf_shared_graph_create(const char* path, const char* signature_key, bool session)
{
	int plen = strlen(path) + 20;

	//reduce logging level of TensorFlow
	if (debug_level <= DP_INFO)
		setenv("TF_CPP_MIN_LOG_LEVEL", "1", false);

	char graph_path[plen];
	int rlen = snprintf(graph_path, plen, "%s.pb", path);
	assert(rlen < plen);

	TF_Status* status = TF_NewStatus();

	TF_Graph* graph;
	TF_Session* sess;
	list_t arg_name_mapping = NULL;
	const char* init_file = NULL;

	FILE *fp = fopen(graph_path, "r");
	if (fp != NULL)
	{
		fclose(fp);

		graph = load_graph(graph_path, status);
		sess = create_session(graph, status);

		if (session)
			restore_session(graph, status, sess, path);
		
		debug_printf(DP_DEBUG1, "Succesfully loaded TensorFlow v1 graph!\n");
	
	} else {

		snprintf(graph_path, plen, "%s/", path);

		graph = TF_NewGraph();

		TF_SessionOptions* sess_opts = get_session_opts();
    	
		TF_Buffer* run_opts = NULL;

		const char* tags = "serve"; // default model serving tag; can change in future
    	int ntags = 1;

		sess = TF_LoadSessionFromSavedModel(sess_opts, run_opts, graph_path, &tags, ntags, graph, NULL, status);

		if (TF_GetCode(status) != TF_OK)
			error("Unable to restore TensorFlow saved model from %s: %s\n", graph_path, TF_Message(status));

		snprintf(graph_path, plen, "%s/bart_config.dat", path);
		arg_name_mapping = read_name_mapping(graph_path, signature_key ?: "serving_default");

		debug_printf(DP_DEBUG1, "Succesfully loaded TensorFlow v2 saved model!\n");

		init_file = ptr_printf("%s/bart_initial_weights", path);
	}	
	
	PTR_ALLOC(struct tf_shared_graph_s, x);

	x->graph = graph;
	x->sess = sess;
	x->status = status;
	x->arg_name_map = arg_name_mapping;
	x->weight_init = init_file;

	shared_obj_init(&x->sptr, tf_shared_graph_del);

	return PTR_PASS(x);

}



static TF_Tensor* tensor_allocate(int N, const long dims[N])
{
	long dims2[N];
	assert(0 < N);

	for (int i = 0; i < N; i++)
		dims2[i] = dims[N - i - 1];

	assert(1 == dims2[N - 1]);
	dims2[N - 1] = 2;

	size_t size = product(N, dims2) * FL_SIZE;

	return TF_AllocateTensor(TF_FLOAT, dims2, N, size);
}
struct tf_s {

	INTERFACE(nlop_data_t);

	int nr_inputs;
	int nr_outputs;

	const struct tf_shared_graph_s* graph;

	TF_Tensor* const* input_tensors;

	struct TF_Output *inputs_op;
	struct TF_Output *outputs_op;
	struct TF_Output *grad_op;

	int *nr_out_dim;
	int *nr_in_dim;

	const int64_t **out_dims_tf;
	const int64_t **in_dims_tf;

	complex float*** cached_gradient;
};

DEF_TYPEID(tf_s);

static void tf_forward(const nlop_data_t* _data, int N, complex float* args[N])
{
	auto data = CAST_DOWN(tf_s, _data);

	assert(data->nr_inputs + data->nr_outputs == N);

	TF_Tensor* output_tensors[data->nr_outputs];

	for (int i = 0; i < data->nr_inputs; i++)
		md_copy(data->nr_in_dim[i], data->in_dims_tf[i], TF_TensorData(data->input_tensors[i]), args[i + data->nr_outputs], CFL_SIZE);

	TF_SessionRun(data->graph->sess,
				/* RunOptions */ NULL,
				/* Input tensors */ data->inputs_op, data->input_tensors, data->nr_inputs + data->nr_outputs,
				/* Output tensors */ data->outputs_op, output_tensors, data->nr_outputs,
				/* Target operations */ NULL, 0,
				/* RunMetadata */ NULL,
				/* Output status */ data->graph->status);

	if (TF_GetCode(data->graph->status) != TF_OK)
		error("Running TensorFlow failed: %s\n", TF_Message(data->graph->status));

	for (int i = 0; i < data->nr_outputs; i++) {

		md_copy(data->nr_out_dim[i], data->out_dims_tf[i], args[i], TF_TensorData(output_tensors[i]), CFL_SIZE);

		TF_DeleteTensor(output_tensors[i]);
	}

	for (int i = 0; i < data->nr_inputs; i++)
		for (int o = 0; o < data->nr_outputs; o++) {
		
			md_free(data->cached_gradient[o][i]);
			data->cached_gradient[o][i] = NULL;
		}
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

	if (   (0 != md_zrmse(data->nr_out_dim[o], data->out_dims_tf[o], TF_TensorData(data->input_tensors[data->nr_inputs + o]), src))
		|| (NULL == data->cached_gradient[o][i])) {
	
		md_copy(data->nr_out_dim[o], data->out_dims_tf[o], TF_TensorData(data->input_tensors[data->nr_inputs + o]), src, CFL_SIZE);

		complex float** grad = data->cached_gradient[o];
		
		int N = 0;
		struct TF_Output grad_ops[data->nr_inputs];

		for (int i = 0; i < data->nr_inputs; i++) {

			md_free(grad[i]);
			grad[i] = NULL;

			if (nlop_der_requested(_data, i, o)) {

				grad[i] = md_alloc(data->nr_in_dim[i], data->in_dims_tf[i], CFL_SIZE);
				grad_ops[N] = data->grad_op[i + data->nr_inputs * o];
				N++;
			}
		}

		struct TF_Tensor* out_tensor[N];

		TF_SessionRun(data->graph->sess,
				/* RunOptions */ NULL,
				/* Input tensors */ data->inputs_op, data->input_tensors, data->nr_inputs + data->nr_outputs,
				/* Output tensors */ grad_ops, out_tensor, N,
				/* Target operations */ NULL, 0,
				/* RunMetadata */ NULL,
				/* Output status */ data->graph->status);

		if (TF_GetCode(data->graph->status) != TF_OK)
			error("Running TensorFlow failed: %s\n", TF_Message(data->graph->status));

		for (int i = 0, ip = 0; i < data->nr_inputs; i++) {

			if (nlop_der_requested(_data, i, o)) {

				md_copy(data->nr_in_dim[i], data->in_dims_tf[i], grad[i], TF_TensorData(out_tensor[ip]), CFL_SIZE);
				TF_DeleteTensor(out_tensor[ip++]);
			}
		}		
	}

	md_copy(data->nr_in_dim[i], data->in_dims_tf[i], dst, data->cached_gradient[o][i], CFL_SIZE);
}


static void tf_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(tf_s, _data);

	for (int i = 0; i < data->nr_inputs + data->nr_outputs; i++)
		TF_DeleteTensor(data->input_tensors[i]);

	tf_shared_graph_free(data->graph);

	xfree(data->input_tensors);

	xfree(data->inputs_op);
	xfree(data->outputs_op);
	xfree(data->grad_op);

	xfree(data->nr_out_dim);
	xfree(data->nr_in_dim);

	for (int i = 0; i < data->nr_inputs; i++)
		xfree(data->in_dims_tf[i]);

	xfree(data->in_dims_tf);

	for (int i = 0; i < data->nr_outputs; i++)
		xfree(data->out_dims_tf[i]);

	for (int o = 0; o < data->nr_outputs; o++) {

		for (int i = 0; i < data->nr_inputs; i++)	
			md_free(data->cached_gradient[o][i]);
		
		xfree(data->cached_gradient[o]);
	}

	xfree(data->cached_gradient);


	xfree(data->out_dims_tf);

	xfree(data);
};



struct tf_arg {

	struct TF_Output out;
	int N;
	const int64_t* dims;
};


static bool cmp_arg_name(const void* _map, const void* _bart_name)
{
	const struct tf_arg_map_s* map = _map;
	const char* bart_name = _bart_name;
	
	return 0 == strcmp(map->bart_name, bart_name);
}

static TF_Output get_output(const struct tf_shared_graph_s* graph, const char* name)
{
	if (NULL != graph->arg_name_map) {

		int idx = list_get_first_index(graph->arg_name_map, name, cmp_arg_name);
		if (-1 == idx)
			return (struct TF_Output){ NULL, 0 };

		const struct tf_arg_map_s* map = list_get_item(graph->arg_name_map, idx);

		return (struct TF_Output){ TF_GraphOperationByName(graph->graph, map->tf_name), map->tf_index };

	}

	return (struct TF_Output){ TF_GraphOperationByName(graph->graph, name), 0 };
}

static bool graph_has_arg(const struct tf_shared_graph_s* graph, const char* name)
{
	return NULL != get_output(graph, name).oper;
}



static struct tf_arg process_arg(const struct tf_shared_graph_s* graph, const char* name)
{
	struct tf_arg arg;

	arg.out = get_output(graph, name);

	if (NULL == arg.out.oper)
		error("Graph operation %s missing.\n", name);

	arg.N = TF_GraphGetTensorNumDims(graph->graph, arg.out, graph->status);

	if (TF_GetCode(graph->status) != TF_OK)
		error("Getting TensorFlow dimensions failed: %s\n", TF_Message(graph->status));

	long tdims[arg.N ?: 1];

	TF_GraphGetTensorShape(graph->graph, arg.out, tdims, arg.N, graph->status);

	if (TF_GetCode(graph->status) != TF_OK)
		error("Getting TensorFlow shape failed: %s\n", TF_Message(graph->status));

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



const struct nlop_s* nlop_tf_shared_create(const struct tf_shared_graph_s* graph)
{
	int II = -1;
	int OO = -1;
	
	char name[20];

	do
		sprintf(name, "input_%d", ++II);
	while (graph_has_arg(graph, name));

	do
		sprintf(name, "output_%d", ++OO);
	while (graph_has_arg(graph, name));
	
	/*** handle outputs and grad_ys ***/

	// outputs
	int ON = 1;
	int ON_arr[OO];

	PTR_ALLOC(struct TF_Output[OO], outputs_op);
	PTR_ALLOC(int[OO], nr_out_dim);
	PTR_ALLOC(const int64_t*[OO], out_dims_tf);

	PTR_ALLOC(struct TF_Output[II + OO], inputs_op);
	PTR_ALLOC(TF_Tensor*[II + OO], input_tensors);

	for (int i = 0; i < OO; i++) {

		char out_name[20];
		sprintf(out_name, "output_%d", i);
		struct tf_arg arg = process_arg(graph, out_name);

		ON_arr[i] = arg.N;
		ON = MAX(ON, ON_arr[i]);

		(*outputs_op)[i] = arg.out;
		(*nr_out_dim)[i] = arg.N;
		(*out_dims_tf)[i] = arg.dims;

		char grad_ys_name[20];
		sprintf(grad_ys_name, "grad_ys_%d", i);

		struct tf_arg arg_grad_y = process_arg(graph, grad_ys_name);

		if (!cmp_arg(arg, arg_grad_y) || (arg.N != arg_grad_y.N))
			error("Tensorflow output and corresponding gradient input do not have the same shape!");

		(*inputs_op)[II + i] = arg_grad_y.out;
		(*input_tensors)[II + i] = tensor_allocate(arg_grad_y.N, arg_grad_y.dims);
		md_clear(arg_grad_y.N, arg_grad_y.dims, TF_TensorData((*input_tensors)[II + i]), CFL_SIZE);

		xfree(arg_grad_y.dims);
	}

	PTR_ALLOC(struct tf_s, data);
	SET_TYPEID(tf_s, data);

	data->graph = tf_shared_graph_ref(graph);
	data->nr_inputs = II;
	data->nr_outputs = OO;

	data->outputs_op = *PTR_PASS(outputs_op);
	data->nr_out_dim = *PTR_PASS(nr_out_dim);
	data->out_dims_tf = *PTR_PASS(out_dims_tf);

	// handle inputs and grad
	int IN = 1;
	int IN_arr[II];

	PTR_ALLOC(int[II], nr_in_dim);
	PTR_ALLOC(const int64_t *[II], in_dims_tf);
	PTR_ALLOC(struct TF_Output[II * OO], grad_op);

	for (int i = 0; i < II; i++) {

		char in_name[20];
		sprintf(in_name, "input_%d", i);

		struct tf_arg arg = process_arg(graph, in_name);

		IN_arr[i] = arg.N;
		IN = MAX(IN, IN_arr[i]);

		(*input_tensors)[i] = tensor_allocate(arg.N, arg.dims);
		(*inputs_op)[i] = arg.out;
		(*nr_in_dim)[i] = arg.N;
		(*in_dims_tf)[i] = arg.dims;


		for (int o = 0; o < OO; o++) {

			char grad_name[30];
			sprintf(grad_name, "grad_%d", i);

			if ((1 != OO) || !graph_has_arg(graph, grad_name))
				sprintf(grad_name, "grad_%d_%d", i, o);

			struct tf_arg arg_grad = process_arg(graph, grad_name);

			if (!cmp_arg(arg, arg_grad))
				error("Tensorflow input and corresponding gradient do not have the same shape!");

			(*grad_op)[i + II * o] = arg_grad.out;

			xfree(arg_grad.dims);
		}
	}

	data->inputs_op = *PTR_PASS(inputs_op);
	data->input_tensors = *PTR_PASS(input_tensors);
	data->nr_in_dim = *PTR_PASS(nr_in_dim);
	data->in_dims_tf = *PTR_PASS(in_dims_tf);
	data->grad_op = *PTR_PASS(grad_op);

	complex float* ci[II];
	for (int i = 0; i < II; i++)
		ci[i] = NULL;
	
	complex float** cached_gradients[OO];
	for (int i = 0; i < OO; i++)
		cached_gradients[i] = ARR_CLONE(complex float*[II], ci);

	data->cached_gradient = ARR_CLONE(complex float**[OO], cached_gradients);	




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

	for (int i = 0; i < II; i++)
		if (1 < IN_arr[i])
			result = nlop_reshape_in_F(result, i, IN_arr[i] - 1, nl_idims[i] + 1);
		else
			result = nlop_reshape_in_F(result, i, 1, MD_DIMS(1));

	for (int i = 0; i < OO; i++)
		if (1 < ON_arr[i])
			result = nlop_reshape_out_F(result, i, ON_arr[i] - 1, nl_odims[i] + 1);
		else
			result = nlop_reshape_out_F(result, i, 1, MD_DIMS(1));

	return result;
}

const struct nlop_s* nlop_tf_create(int OO, int II, const char* path, bool session)
{
	const struct tf_shared_graph_s* graph = tf_shared_graph_create(path, NULL, session);

	const struct nlop_s* result = nlop_tf_shared_create(graph);

	tf_shared_graph_free(graph);
	
	assert((-1 == II) || (II == nlop_get_nr_in_args(result)));
	assert((-1 == OO) || (OO == nlop_get_nr_out_args(result)));

	return result;
}

#endif




