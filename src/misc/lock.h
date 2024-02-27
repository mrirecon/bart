
typedef struct bart_lock bart_lock_t;

void bart_lock(bart_lock_t* lock);
void bart_unlock(bart_lock_t* lock);
void bart_lock_destroy(bart_lock_t* x);
bart_lock_t* bart_lock_create(void);

