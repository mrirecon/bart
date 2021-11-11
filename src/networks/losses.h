struct loss_config_s {

	float epsilon;

	float weighting_mse;
	float weighting_mad;
	float weighting_mse_rss;
	float weighting_mad_rss;
	float weighting_psnr_rss;
	float weighting_ssim_rss;
	float weighting_nmse;
	float weighting_nmse_rss;


	float weighting_cce;
	float weighting_weighted_cce;
	float weighting_accuracy;

	float weighting_dice0;
	float weighting_dice1;
	float weighting_dice2;

	float weighting_dice_labels;

	int label_index;
	unsigned long image_flags;
	unsigned long rss_flags;
	unsigned long mse_mean_flags;

	unsigned long mask_flags;
};

extern struct loss_config_s val_loss_option;
extern struct loss_config_s loss_option;

extern struct loss_config_s loss_image_valid;
extern struct loss_config_s loss_classification_valid;

extern const struct nn_s* train_loss_create(const struct loss_config_s* config, unsigned int N, const long dims[N]);
extern const struct nn_s* val_measure_create(const struct loss_config_s* config, unsigned int N, const long dims[N]);