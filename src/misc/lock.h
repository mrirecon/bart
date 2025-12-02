
typedef struct bart_lock bart_lock_t;

extern void bart_lock(bart_lock_t* lock);
extern _Bool bart_trylock(bart_lock_t* lock);
extern void bart_unlock(bart_lock_t* lock);
extern void bart_lock_destroy(bart_lock_t* x);
extern bart_lock_t* bart_lock_create(void);


typedef struct bart_cond bart_cond_t;

extern void bart_cond_wait(bart_cond_t* cond, bart_lock_t* lock);
extern void bart_cond_notify_all(bart_cond_t* cond);
extern void bart_cond_destroy(bart_cond_t* x);
extern bart_cond_t* bart_cond_create(void);




