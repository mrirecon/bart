#include "misc/graph.h"

struct operator_s;
struct iovec_s;

typedef struct node_s* node_t;
typedef const struct graph_s* graph_t;

struct list_s;
typedef struct list_s* list_t;

extern graph_t create_graph_operator(const struct operator_s* op, const char* name);
extern graph_t create_graph_container(const struct operator_s* op, const char* name, graph_t subgraph);

extern graph_t operator_graph_combine_F(int N, graph_t ops[N]);
extern graph_t operator_graph_chain_F(int N, graph_t ops[N]);
extern graph_t operator_graph_dup_F(graph_t op, int a, int b);
extern graph_t operator_graph_link_F(graph_t op, int oo, int ii);
extern graph_t operator_graph_permute_F(graph_t op, int N, const int perm[N]);
extern graph_t operator_graph_reshape_F(graph_t op, int i, int N, const long dims[N]);

extern void operator_export_graph_dot(const char* filename, const struct operator_s* op);

extern const struct operator_s* graph_to_operator_F(graph_t _graph);

extern graph_t operator_graph_optimize_chains_F(graph_t graph);
extern graph_t operator_graph_optimize_clusters_F(graph_t graph, _Bool simple_only);
extern graph_t operator_graph_optimize_identity_F(graph_t graph);
extern graph_t operator_graph_optimize_identify_F(graph_t graph);
extern graph_t operator_graph_optimize_linops_F(graph_t graph, node_cmp_t linop_identify);
extern graph_t operator_graph_sum_to_multi_sum_F(graph_t graph, _Bool inverse);

extern const struct operator_s* get_operator_from_node(const struct node_s* _node);