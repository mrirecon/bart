/* Copyright 2021. Uecker Lab. University Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>

#include "misc/debug.h"
#include "misc/opts.h"
#include "iter/iter6.h"

#include "opt_iter6.h"

struct iter6_conf_s iter6_conf_unset = {

	.learning_rate = 0.,
	.epochs = -1,
	.clip_norm = 0.,
	.clip_val = 0.,
	.weight_decay = 0.,
	.history_filename = NULL,
	.dump_filename = NULL,
	.dump_mod = -1,
	.batchnorm_momentum = .95,
	.batchgen_type = BATCH_GEN_SAME,
	.batch_seed = 123,
	.dump_flag = 0,
	.monitor_averaged_objective = false,
};

struct iter6_conf_s iter6_conf_opts = {

	.learning_rate = 0.,
	.epochs = -1,
	.clip_norm = 0.,
	.clip_val = 0.,
	.weight_decay = 0.,
	.history_filename = NULL,
	.dump_filename = NULL,
	.dump_mod = -1,
	.batchnorm_momentum = .95,
	.batchgen_type = BATCH_GEN_SAME,
	.batch_seed = 123,
	.dump_flag = 0,
	.monitor_averaged_objective = false,
};

struct iter6_sgd_conf iter6_sgd_conf_opts;
struct iter6_adadelta_conf iter6_adadelta_conf_opts;
struct iter6_adam_conf iter6_adam_conf_opts;
struct iter6_iPALM_conf iter6_iPALM_conf_opts;
static bool confs_init = false;

void opts_iter6_init(void)
{
	confs_init = true;
	iter6_sgd_conf_opts = iter6_sgd_conf_defaults;
	iter6_adadelta_conf_opts = iter6_adadelta_conf_defaults;
	iter6_adam_conf_opts = iter6_adam_conf_defaults;
	iter6_iPALM_conf_opts = iter6_iPALM_conf_defaults;
}


enum ITER6_TRAIN_ALGORITHM iter_6_select_algo = ITER6_NONE;

struct opt_s iter6_opts[] = {

	OPTL_FLOAT('r', "learning-rate", &(iter6_conf_opts.learning_rate), "f", "learning rate"),
	OPTL_INT('e', "epochs", &(iter6_conf_opts.epochs), "d", "number of epochs to train"),

	OPTL_SELECT_DEF(0, "sgd", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_SGD, ITER6_NONE, "select stochastic gradient descent"),
	OPTL_SELECT_DEF(0, "adadelta", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_ADADELTA, ITER6_NONE, "select AdaDelta"),
	OPTL_SELECT_DEF(0, "adam", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_ADAM, ITER6_NONE, "select Adam"),
	OPTL_SELECT_DEF(0, "ipalm", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_IPALM, ITER6_NONE, "select iPALM"),

	OPTL_FLOAT(0, "clip-norm", &(iter6_conf_opts.clip_norm), "f", "clip norm of gradients"),
	OPTL_FLOAT(0, "clip-value", &(iter6_conf_opts.clip_val), "f", "clip value of gradients"),

	OPTL_FLOAT(0, "weight-decay", &(iter6_conf_opts.weight_decay), "f", "reduce weights by 1 / (1 + lr*f) after each update"),

	OPTL_INT(0, "epochs-warmup", &(iter6_conf_opts.epochs_warmup), "d", "linearly increase learning rate in first d epochs"),
	OPTL_FLOAT(0, "learning-rate-min", &(iter6_conf_opts.min_learning_rate), "f", "minimum learning rate (cosine annealing / exponential decay)"),
	OPTL_INT(0, "cosine-annealing-mod", &(iter6_conf_opts.learning_rate_epoch_mod), "d", "schedule learning rate using cosine annealing each d epochs"),

	OPTL_OUTFILE(0, "export-history", &(iter6_conf_opts.history_filename), "<file>", "export file containing the train history"),

	OPTL_LONG(0, "dump-mod", &(iter6_conf_opts.dump_mod), "mod", "dump weights to file every \"mod\" epochs"),

	OPTL_FLOAT(0, "batchnorm-momentum", &(iter6_conf_opts.batchnorm_momentum), "f", "momentum for batch normalization (default: 0.95)"),

	OPTL_SELECT_DEF(0, "batchgen-same", enum BATCH_GEN_TYPE, &(iter6_conf_opts.batchgen_type), BATCH_GEN_SAME, BATCH_GEN_SAME, "use the same batches in the same order for each epoch"),
	OPTL_SELECT_DEF(0, "batchgen-shuffle-batches", enum BATCH_GEN_TYPE, &(iter6_conf_opts.batchgen_type), BATCH_GEN_SHUFFLE_BATCHES, BATCH_GEN_SAME, "use the same batches in random order for each epoch"),
	OPTL_SELECT_DEF(0, "batchgen-shuffle-data", enum BATCH_GEN_TYPE, &(iter6_conf_opts.batchgen_type), BATCH_GEN_SHUFFLE_DATA, BATCH_GEN_SAME, "shuffle data to form batches"),
	OPTL_SELECT_DEF(0, "batchgen-draw-data", enum BATCH_GEN_TYPE, &(iter6_conf_opts.batchgen_type), BATCH_GEN_RANDOM_DATA, BATCH_GEN_SAME, "randomly draw data to form batches"),
	OPTL_INT(0, "batchgen-seed", &(iter6_conf_opts.batch_seed), "d", "seed for batch-generator (default: 123)"),
	OPTL_SET(0, "average-loss", &(iter6_conf_opts.monitor_averaged_objective), "monitor loss averaged over epoch"),
};

struct opt_s iter6_sgd_opts[] = {

	OPTL_SELECT_DEF('s', "sgd", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_SGD, ITER6_NONE, "select stochastic gradient descent"),
	OPTL_FLOAT(0, "momentum", &(iter6_sgd_conf_opts.momentum), "f", "momentum (default: 0.)"),
};
const int N_iter6_sgd_opts = ARRAY_SIZE(iter6_sgd_opts);

struct opt_s iter6_adadelta_opts[] = {

	OPTL_SELECT_DEF('s', "adadelta", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_ADADELTA, ITER6_NONE, "select AdaDelta"),
	OPTL_FLOAT(0, "rho", &(iter6_adadelta_conf_opts.rho), "f", "rho (default: 0.95"),
};

struct opt_s iter6_adam_opts[] = {

	OPTL_SELECT_DEF('s', "adam", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_ADAM, ITER6_NONE, "select Adam"),
	OPTL_FLOAT(0, "epsilon", &(iter6_adam_conf_opts.epsilon), "f", "epsilon (default: 1.e-7)"),
	OPTL_FLOAT(0, "beta1", &(iter6_adam_conf_opts.beta1), "f", "beta1 (default: 0.9)"),
	OPTL_FLOAT(0, "beta2", &(iter6_adam_conf_opts.beta2), "f", "beta2 (default: 0.999)"),

	OPTL_LONG(0, "reset-momentum", &(iter6_adam_conf_opts.reset_epoch), "d", "reset momentum every nth epoch (default: -1=never)"),
};

struct opt_s iter6_ipalm_opts[] = {

	OPTL_SELECT_DEF('s', "ipalm", enum ITER6_TRAIN_ALGORITHM, &(iter_6_select_algo), ITER6_IPALM, ITER6_NONE, "select iPALM"),

	OPTL_FLOAT(0, "L-min", &(iter6_iPALM_conf_opts.Lmin), "f", "minimum Lipshitz constant for backtracking (default: 1.e-10)"),
	OPTL_FLOAT(0, "L-max", &(iter6_iPALM_conf_opts.Lmax), "f", "maximum Lipshitz constant for backtracking (default: 1.e10)"),
	OPTL_FLOAT(0, "L-reduce", &(iter6_iPALM_conf_opts.Lshrink), "f", "factor to reduce Lipshitz constant in backtracking (default: 1.2)"),
	OPTL_FLOAT(0, "L-increase", &(iter6_iPALM_conf_opts.Lincrease), "f", "factor to increase Lipshitz constant in backtracking (default: 2)"),

	OPTL_FLOAT(0, "alpha", &(iter6_iPALM_conf_opts.alpha), "f", "alpha factor (default: -1. = \"dynamic case\")"),
	OPTL_FLOAT(0, "beta", &(iter6_iPALM_conf_opts.beta), "f", "beta factor (default: -1. = \"dynamic case\")"),
	OPTL_SET(0, "convex", &(iter6_iPALM_conf_opts.convex), "convex constraints"),

	OPTL_CLEAR(0, "non-trivial-step-size", &(iter6_iPALM_conf_opts.trivial_stepsize), "set stepsize based on alpha and beta"),
	OPTL_CLEAR(0, "no-momentum-reduction", &(iter6_iPALM_conf_opts.reduce_momentum), "backtracking only reduces stepsize not momentum"),
};

const int N_iter6_opts = ARRAY_SIZE(iter6_opts);
const int N_iter6_adadelta_opts = ARRAY_SIZE(iter6_adadelta_opts);
const int N_iter6_adam_opts = ARRAY_SIZE(iter6_adam_opts);
const int N_iter6_ipalm_opts = ARRAY_SIZE(iter6_ipalm_opts);

void iter6_copy_config_from_opts(struct iter6_conf_s* result)
{
	if (iter6_conf_opts.learning_rate != iter6_conf_unset.learning_rate)
		result->learning_rate = iter6_conf_opts.learning_rate;
	if (iter6_conf_opts.epochs != iter6_conf_unset.epochs)
		result->epochs = iter6_conf_opts.epochs;
	if (iter6_conf_opts.clip_norm != iter6_conf_unset.clip_norm)
		result->clip_norm = iter6_conf_opts.clip_norm;
	if (iter6_conf_opts.clip_val != iter6_conf_unset.clip_val)
		result->clip_val = iter6_conf_opts.clip_val;
	if (iter6_conf_opts.weight_decay != iter6_conf_unset.weight_decay)
		result->weight_decay = iter6_conf_opts.weight_decay;
	if (iter6_conf_opts.history_filename != iter6_conf_unset.history_filename)
		result->history_filename = iter6_conf_opts.history_filename;
	if (iter6_conf_opts.dump_filename != iter6_conf_unset.dump_filename)
		result->dump_filename = iter6_conf_opts.dump_filename;
	if (iter6_conf_opts.dump_mod != iter6_conf_unset.dump_mod)
		result->dump_mod = iter6_conf_opts.dump_mod;
	if (iter6_conf_opts.batchnorm_momentum != iter6_conf_unset.batchnorm_momentum)
		result->batchnorm_momentum = iter6_conf_opts.batchnorm_momentum;
	if (iter6_conf_opts.batchgen_type != iter6_conf_unset.batchgen_type)
		result->batchgen_type = iter6_conf_opts.batchgen_type;
	if (iter6_conf_opts.batch_seed != iter6_conf_unset.batch_seed)
		result->batch_seed = iter6_conf_opts.batch_seed;
	if (iter6_conf_opts.learning_rate_epoch_mod != iter6_conf_unset.learning_rate_epoch_mod)
		result->learning_rate_epoch_mod = iter6_conf_opts.learning_rate_epoch_mod;
	if (iter6_conf_opts.min_learning_rate != iter6_conf_unset.min_learning_rate)
		result->min_learning_rate = iter6_conf_opts.min_learning_rate;
	if (iter6_conf_opts.epochs_warmup != iter6_conf_unset.epochs_warmup)
		result->epochs_warmup = iter6_conf_opts.epochs_warmup;
	if (iter6_conf_opts.monitor_averaged_objective != iter6_conf_unset.monitor_averaged_objective)
		result->monitor_averaged_objective = iter6_conf_opts.monitor_averaged_objective;
}

struct iter6_conf_s* iter6_get_conf_from_opts(void)
{
	assert(confs_init);

	struct iter6_conf_s* result = NULL;

	switch (iter_6_select_algo) {

	case ITER6_NONE:
		debug_printf(DP_INFO, "No training algorithm selected! Fallback to default settings.\n");
		return result;
		break;

	case ITER6_SGD:
		result = CAST_UP(&iter6_sgd_conf_opts);
		break;

	case ITER6_ADAM:
		result = CAST_UP(&iter6_adam_conf_opts);
		break;

	case ITER6_ADADELTA:
		result = CAST_UP(&iter6_adam_conf_opts);
		break;

	case ITER6_IPALM:
		result = CAST_UP(&iter6_adam_conf_opts);
		break;
	}

	iter6_copy_config_from_opts(result);

	return result;
}

