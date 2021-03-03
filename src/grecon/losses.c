#include "misc/opts.h"

#include "networks/losses.h"
#include "losses.h"

struct opt_s loss_opts[] = {

	OPTL_FLOAT(0, "mse", &(loss_option.weighting_mse), "weighting", "weighting for mean squared error"),
	OPTL_FLOAT(0, "mse-magnitude", &(loss_option.weighting_mse_sa), "weighting", "weighting for smoothed mean squared error of magnitude"),
	OPTL_FLOAT(0, "ssim", &(loss_option.weighting_ssim), "weighting", "weighting for structural similarity index measure (magnitude)"),

	OPTL_FLOAT(0, "cce", &(loss_option.weighting_cce), "weighting", "weighting for categorical cross entropy"),
	OPTL_FLOAT(0, "wcce", &(loss_option.weighting_weighted_cce), "weighting", "weighting for weighted categorical cross entropy"),

	OPTL_FLOAT(0, "dice-0", &(loss_option.weighting_dice0), "weighting", "weighting for unbalanced dice loss"),
	OPTL_FLOAT(0, "dice-1", &(loss_option.weighting_dice1), "weighting", "weighting for dice loss weighted with inverse frequency of label"),
	OPTL_FLOAT(0, "dice-2", &(loss_option.weighting_dice2), "weighting", "weighting for dice loss weighted with inverse square frequency of label"),
};

const int N_loss_opts = ARRAY_SIZE(loss_opts);


struct opt_s val_loss_opts[] = {

	OPTL_FLOAT(0, "mse", &(val_loss_option.weighting_mse), "weighting", "weighting for mean squared error"),
	OPTL_FLOAT(0, "mse-magnitude", &(val_loss_option.weighting_mse_sa), "weighting", "weighting for smoothed mean squared error of magnitude"),
	OPTL_FLOAT(0, "psnr", &(val_loss_option.weighting_psnr), "weighting", "weighting for peak signal to noise ratio (no training)"),
	OPTL_FLOAT(0, "ssim", &(val_loss_option.weighting_ssim), "weighting", "weighting for structural similarity index measure"),

	OPTL_FLOAT(0, "cce", &(val_loss_option.weighting_cce), "weighting", "weighting for categorical cross entropy"),
	OPTL_FLOAT(0, "wcce", &(val_loss_option.weighting_weighted_cce), "weighting", "weighting for weighted categorical cross entropy"),
	OPTL_FLOAT(0, "acc", &(val_loss_option.weighting_accuracy), "weighting", "weighting for accuracy (no training)"),

	OPTL_FLOAT(0, "dice-0", &(val_loss_option.weighting_dice0), "weighting", "weighting for unbalanced dice loss"),
	OPTL_FLOAT(0, "dice-1", &(val_loss_option.weighting_dice1), "weighting", "weighting for dice loss weighted with inverse frequency of label"),
	OPTL_FLOAT(0, "dice-2", &(val_loss_option.weighting_dice2), "weighting", "weighting for dice loss weighted with inverse square frequency of label"),

	OPTL_FLOAT(0, "dice-l", &(val_loss_option.weighting_dice_labels), "weighting", "weighting for per label dice loss"),
};

const int N_val_loss_opts = ARRAY_SIZE(val_loss_opts);

static bool loss_option_changed(struct loss_config_s* loss_option)
{
	if (0 != loss_option->weighting_mse_sa)
		return true;
	if (0 != loss_option->weighting_mse)
		return true;
	if (0 != loss_option->weighting_psnr)
		return true;
	if (0 != loss_option->weighting_ssim)
		return true;

	if (0 != loss_option->weighting_cce)
		return true;
	if (0 != loss_option->weighting_weighted_cce)
		return true;
	if (0 != loss_option->weighting_accuracy)
		return true;

	if (0 != loss_option->weighting_dice0)
		return true;
	if (0 != loss_option->weighting_dice1)
		return true;
	if (0 != loss_option->weighting_dice2)
		return true;

	return false;
}

struct loss_config_s* get_loss_from_option(void)
{
	if (!loss_option_changed(&loss_option))
		return NULL;

	return &loss_option;
}

struct loss_config_s* get_val_loss_from_option(void)
{
	if (!loss_option_changed(&val_loss_option))
		return NULL;

	return &val_loss_option;
}