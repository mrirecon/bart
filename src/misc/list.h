
struct list_s;
typedef struct list_s* list_t;


extern list_t list_create(void);
extern void list_free(list_t list);

extern int list_count(list_t list);

extern void list_push(list_t list, void* item);
extern void list_append(list_t list, void* item);

extern void* list_pop(list_t list);
extern void* list_remove_item(list_t list, int index);
extern void* list_get_item(list_t list, int index);

extern void list_to_array(int N, void* items[N], list_t list);
extern list_t array_to_list(int N, void* items[N]);

typedef _Bool (*list_cmp_t)(const void* item, const void* ref);

extern list_t list_get_sublist(list_t list, const void* ref, list_cmp_t cmp);
extern list_t list_pop_sublist(list_t list, const void* ref, list_cmp_t cmp);

extern int list_get_first_index(list_t list, const void* ref, list_cmp_t cmp);
extern int list_count_cmp(list_t list, const void* ref, list_cmp_t cmp);

extern void list_merge(list_t a, list_t b, _Bool free);
extern list_t list_copy(list_t list);
