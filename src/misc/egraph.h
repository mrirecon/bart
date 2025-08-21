
#ifndef EGRAPH_H
#define EGRAPH_H

#include "misc/cppwrap.h"

struct enode_s;
typedef struct enode_s* enode_t;

struct list_s;
typedef struct list_s* list_t;

void enode_free(enode_t x);
enode_t enode_create(const char* name, const void* data);

extern _Bool enode_is_active(enode_t node);
extern void* enode_get_data(enode_t node);
extern long enode_get_count(enode_t node);

extern void enode_add_dependency(enode_t a, enode_t b);
extern list_t enode_get_iedges(enode_t node);
extern list_t enode_get_oedges(enode_t node);

typedef struct list_s* egraph_t;
extern egraph_t egraph_create(void);
extern void egraph_free(egraph_t graph);
extern void egraph_add_node(egraph_t graph, enode_t node);
extern enode_t egraph_get_node(egraph_t graph, int idx);
extern enode_t egraph_remove_node(egraph_t graph, int idx);

extern long egraph_diameter(egraph_t graph);


extern long egraph_depth_first_search(egraph_t graph, enode_t src, long count, _Bool reverse);

extern void egraph_dijkstra(egraph_t graph, enode_t src, _Bool reverse);
extern void egraph_bfs(egraph_t graph, enode_t src, _Bool reverse);

extern list_t egraph_shortest_path(egraph_t graph, enode_t src, enode_t dst);
extern long egraph_longest_distance(enode_t* dst, enode_t* src, egraph_t graph, list_t nodes);

extern enode_t egraph_find_most_distant(egraph_t graph, enode_t src);
extern list_t egraph_split_connected_components(egraph_t graph);

extern void egraph_topological_sort_F(list_t graph);
extern _Bool enode_depends_on(list_t graph, enode_t a, enode_t b);

extern void egraph_set_ancestors(enode_t b);
extern void egraph_set_descendants(enode_t a);
extern void egraph_reset_between(list_t graph);
extern list_t egraph_sort_between(list_t graph);
extern _Bool enode_is_between(enode_t node);
extern _Bool enode_is_ancestors(enode_t b);
extern _Bool enode_is_descendants(enode_t a);


extern void egraph_set_active(list_t graph);
extern void egraph_unset_active(list_t graph);

void export_egraph_dot(const char* filename, list_t graph);


#include "misc/cppwrap.h"

#endif
