

extern void memcache_init(void);
extern void memcache_destroy(void);
extern void memcache_off(void);

extern void* mem_device_malloc(long size, void* (*device_alloc)(size_t));
extern void mem_device_free(void* ptr, void (*device_free)(const void* x));
extern void memcache_clear(void (*device_free)(const void* x));

extern _Bool mem_ondevice(const void* ptr);

extern void debug_print_memcache(int dl);
extern _Bool memcache_is_empty(void);

