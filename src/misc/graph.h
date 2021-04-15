
#include "misc/types.h"
#include "misc/shrdptr.h"

struct node_s;
struct graph_s;
struct vertex_s;

typedef void (*node_del_t)(const struct node_s*);

typedef struct node_s* node_t;
typedef const struct graph_s* graph_t;

struct list_s;
typedef struct list_s* list_t;

typedef void (*node_del_t)(const struct node_s*);
typedef const char* (*node_print_t)(const struct node_s*);
typedef void (*edge_separator_node_f)(node_t ext_nodes[2], struct vertex_s);

struct node_s {

	TYPEID* TYPEID;

	_Bool external;
	int N_vertices;
	list_t* edges;
	_Bool* io_flags;

	long count;

	const char* name;
	graph_t subgraph;

	node_print_t node_print;
	node_del_t node_del;
};

struct graph_s {

	list_t nodes;
	list_t ext_nodes;
};

struct vertex_s {

	node_t node;
	int idx;
};

typedef struct vertex_s* vertex_t;

void node_free(node_t x);
void node_init(struct node_s* x, int N_vertices, const _Bool io_flags[N_vertices], const char* name, _Bool external, graph_t subgraph);

void graph_free(graph_t x);
graph_t graph_create(void);

extern void graph_add_node(graph_t graph, node_t node);
extern void graph_add_edge(struct vertex_s _a, struct vertex_s _b);
extern void graph_remove_node(graph_t graph, node_t node);
extern void graph_remove_edge(struct vertex_s a, struct vertex_s b);

extern void graph_redirect_edge(struct vertex_s new, struct vertex_s old);

extern graph_t copy_graph(graph_t graph);
extern graph_t combine_graphs_F(int N, graph_t graphs[N]);
extern graph_t link_graphs_F(graph_t graph, int oo, int ii);
extern graph_t perm_ext_graphs_F(graph_t graph, int N, const int perm[N]);
extern graph_t dup_graphs_F(graph_t graph, int a, int b);

extern const char* print_vertex(node_t node, int idx);
extern const char* print_node(const struct node_s* node);
extern const char* print_internl_graph(graph_t graph, _Bool get_ext_nodes, int N, const char* ext_nodes[N]);
extern void export_graph_dot(const char* filename, graph_t graph);
