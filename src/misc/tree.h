
struct tree_s;
typedef struct tree_s* tree_t;

// order relation:
// tree_rel_f(a, b) =  1  if a > b
// tree_rel_f(a, b) =  0  if a = b
// tree_rel_f(a, b) = -1  if a < b

typedef int (*tree_rel_f)(const void* a, const void* b);

#include "misc/cppwrap.h"

// This implements an AVL tree
extern tree_t tree_create(tree_rel_f rel);
extern void tree_free(tree_t tree);

extern int tree_count(tree_t tree);
extern void tree_to_array(tree_t tree, int N, void* arr[__VLA(N)]);

extern void tree_insert(tree_t tree, void* item);
extern void ptr_tree_insert(tree_t tree, void *item, const void* key, unsigned long len);

extern void* tree_find_min(tree_t tree, const void* ref, tree_rel_f rel, _Bool remove);
extern void* tree_find_max(tree_t tree, const void* ref, tree_rel_f rel, _Bool remove);
extern void* tree_find(tree_t tree, const void* ref, tree_rel_f rel, _Bool remove);

extern void* tree_get_min(tree_t tree, _Bool remove);
extern void* tree_get_max(tree_t tree, _Bool remove);

extern void* tree_tag_ptr(tree_t tree, void* ptr);
extern void* tree_untag_ptr(tree_t tree, void* ptr);
extern int tree_get_tag(tree_t tree, const void* ptr);

#include "misc/cppwrap.h"
