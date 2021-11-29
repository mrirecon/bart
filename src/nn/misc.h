
extern void onehotenc_to_index(unsigned int N, const long odims[N], _Complex float* dst, const long idims[N], const _Complex float*src);
extern void index_to_onehotenc(unsigned int N, const long odims[N], _Complex float* dst, const long idims[N], const _Complex float*src);

extern void onehotenc_set_max_to_one(unsigned int N, const long dims[N], unsigned int class_index, _Complex float* dst, const _Complex float* src);
extern float onehotenc_accuracy(unsigned int N, const long dims[N], unsigned int class_index, const _Complex float* cmp, const _Complex float* ref);

extern void onehotenc_confusion_matrix(unsigned int N, const long dims[N], unsigned int class_index, _Complex float* dst, const _Complex float* pred, const _Complex float* ref);
extern void print_confusion_matrix(unsigned int N, const long dims[N], unsigned int class_index, const _Complex float* pred, const _Complex float* ref);