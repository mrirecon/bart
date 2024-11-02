
#ifndef EGRAPH_H
#define EGRAPH_H

#include "misc/cppwrap.h"

struct enode_s;
struct egraph_s;


typedef struct enode_s* enode_t;

struct list_s;
typedef struct list_s* list_t;

struct enode_s {

	list_t iedges;
	list_t oedges;

	unsigned long flags;

	long count;
	_Bool active;

	const char* name;
	const void* data;
};

void enode_free(enode_t x);
enode_t enode_create(const char* name, const void* data);

extern void enode_add_dependency(enode_t a, enode_t b);
extern void egraph_topological_sort_F(list_t graph);
extern _Bool enode_depends_on(list_t graph, enode_t a, enode_t b);

extern void egraph_set_ancestors(enode_t b);
extern void egraph_set_descendants(enode_t a);
extern void egraph_reset_between(list_t graph);
extern void egraph_sort_between(list_t graph);
extern void subgraph_between(list_t graph, enode_t a, enode_t b);

extern void egraph_set_active(list_t graph);
extern void egraph_unset_active(list_t graph);

void export_egraph_dot(const char* filename, list_t graph);


#include "misc/cppwrap.h"

#endif
