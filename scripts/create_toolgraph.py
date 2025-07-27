#!/usr/bin/env python3
import os
import re

def parse_hdr(fname):
    ret = {}
    key = None
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith("# "):
                key = line[2:].strip()
            elif not key is None:
                ret[key] = line.strip()
    return ret

def short_file_name(fname):
    fname = fname.split("/")[-1]
    ending = fname.split(".")[-1]
    if ending in ["hdr"]:
        cut = -1 * (len(ending) + 1)
        fname = fname[:cut]
    return fname


start_level = 1
colorscheme_max_level = 7

def level_dot(level):
    return f"fillcolor={level if level <= colorscheme_max_level else 'gray'},style=filled"

def File_node(tool_node, hdr, level=start_level):
    name = short_file_name(hdr)
    node_id = f"{name}_{tool_node['id'] if tool_node else 'ROOT'}"
    ftype = 'pipe' if hdr.endswith(".fifo") or hdr == '-' else 'file'
    return {'id': node_id, 'type': ftype, 'level': level,
            'dot': f" [label=\"{name}\",shape=diamond,{level_dot(level)}]" }

def Tool_node(node_id, level=start_level):
    if node_id is None:
        return None
    name = node_id.split("_")[0]
    d = { 'id': node_id, 'type': 'tool','src': [], 'level': level, 'sync' : False,
         'dot': f" [label=\"{name}\",{level_dot(level)}]" }
    try:
        info = parse_hdr(os.path.join(node_dir, node_id))
        for src in info.get("Input Nodes", "").split():
            x = src.split(":")
            d['src'].append((x[0], x[1], x[1].endswith(".fifo") or x[1] == '-'))
        d['command'] = info.get("Command")
        d['directory'] = info.get("Directory")
    except FileNotFoundError:
        d['deleted'] = True
    return d

def Edge(node, src, name=None):
    key = f"{src['id']}->{node['id']}"
    label = ""
    if name and name != "-":
        label = f" [label=\"{short_file_name(name)}\"]"
    return { key:  {'dot': key + label } }


class Graph:
    def __init__(self, start_file, node_dir):
        self.edges = {}
        self.nodes = {}
        self.src_nodes = {}
        self.desc_nodes = {}

        self._make_graph(start_file, node_dir)

    def add_edge(self, node, src, name=None):
        self.edges.update(Edge(node, src, name))
        if not node['id'] in self.src_nodes:
            self.src_nodes[node['id']] = {}
        if not src['id'] in self.desc_nodes:
            self.desc_nodes[src['id']] = {}
        self.src_nodes[node['id']][src['id']] = name
        self.desc_nodes[src['id']][node['id']] = name

    def add_node(self, node):
        self.nodes[node['id']] = node

    def node_has_stdin(self, node):
        return "-" in self.src_nodes.get(node['id'], {}).values()

    def node_has_stdout(self, node):
        return "-" in self.desc_nodes.get(node['id'], {}).values()

    def _step(graph, node):
        graph.add_node(node)
        src_nodes = []
        for src_node_id, src_name, src_is_piped in node['src']:
            level = node['level'] + (0 if src_is_piped else 1)
            src_node = Tool_node(src_node_id, level)
            if src_is_piped:
                graph.add_edge(node, src_node, src_name)
            else:
                src_node['sync'] = True
                file_node = File_node(src_node, src_name, level)
                graph.add_node(file_node)
                graph.add_edge(file_node, src_node)
                graph.add_edge(node, file_node)

            if src_node_id not in graph.nodes or level > graph.nodes[src_node_id]['level']:
                src_nodes.append(src_node)

        return src_nodes

    def _make_graph(graph, start_file, node_dir):
        graph.start_node = Tool_node(parse_hdr(start_file).get("Node-ID"))
        graph.start_node['sync'] = True
        file_node = File_node(graph.start_node, start_file)

        graph.add_node(file_node)
        if graph.start_node:
            graph.add_edge(file_node, graph.start_node)
        src_nodes = [graph.start_node] if graph.start_node else []

        while len(src_nodes) > 0:
            new_src_nodes = []
            for node in src_nodes:
                new_src_nodes += graph._step(node)
            src_nodes = new_src_nodes

    def get_dot(graph):
        ret = "digraph {\n\tnode [colorscheme=" + f"greens{colorscheme_max_level}" + "];\n"
        for node_id, node_opts in graph.nodes.items():
            ret += f"\t{node_id}{node_opts['dot']};\n"
        for edge in graph.edges.values():
            ret += f"\t{edge['dot']};\n"
        ret += "}\n"
        return ret

    def get_heads(self):
        heads = set()
        srcs = self.src_nodes[self.start_node['id']]
        while len(srcs) > 0:
            new_srcs = {}
            for src in srcs:
                x = self.src_nodes.get(src, {})
                if len(x) == 0:
                    heads.add(src)
                new_srcs.update(x)
            srcs = new_srcs
        return list(heads)

    def create_cmds(self, filter_path = None):
        heads = self.get_heads()
        max_level = 1
        for node_id in heads:
            max_level = max(max_level, self.nodes[node_id]['level'])

        level = max_level
        cmds = []
        done = {}
        while level > 0:
            new_cmds = []
            updates = 1
            strands = []
            while updates:
                updates = 0
                new_heads = []
                for node_id in heads:
                    strand = []
                    while node_id:
                        node = self.nodes[node_id]
                        desc = list(self.desc_nodes.get(node['id'], {}).keys())

                        if node['type'] == 'file':
                            new_heads += desc
                            break;
                        if node_id in done:
                            break;
                        if node['level'] != level:
                            new_heads += [node_id]
                            break

                        updates += 1
                        strand.append(node)
                        done[node_id] = 1

                        node_id = None

                        if len(desc) == 1 and not (self.node_has_stdin(self.nodes[desc[0]]) and (self.src_nodes[desc[0]][node['id']] != "-")):
                            node_id = desc[0]
                        else:
                            for desc_id, name in self.desc_nodes[node['id']].items():
                                # continue along stdout
                                if name == "-":
                                    node_id = desc_id
                                else:
                                    new_heads.append(desc_id)
                    strands.append(strand)

                heads = new_heads
            for strand in sorted(strands, key = lambda x: len(x) > 0 and x[-1]['sync']):
                cmds.append(strand)
            level -= 1

        if filter_path:
            rex = re.compile(filter_path)

        ret = "#!/bin/bash\nexport BASEDIR=$(realpath .)\n"
        d = ""
        last_d = None
        e = ""
        t = ""
        for strand in cmds:
            for i, cmd in enumerate(strand):
                last_d = d
                d = cmd['directory']
                if filter_path:
                    d = rex.sub('', d)
                if last_d != d:
                    t = '\t'
                    ret += f"{e}(\n{t}mkdir -p {d}; cd {d};\n"
                    e = ");\n"

                if cmd['sync']:
                    sep = ";"
                elif self.node_has_stdout(cmd):
                    if i + 1 < len(strand) and strand[i + 1]['directory'] == d:
                        sep = " |\\"
                    else:
                        sep = " &"
                        e = ") |\\\n"
                else:
                    sep = " &"

                c = cmd['command']
                if filter_path:
                    c = rex.sub('$BASEDIR/', c)
                ret += t + c + sep  + '\n'
        return ret + e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help=".hdr-file for which to generate graph")
    parser.add_argument("--type", choices=['dot', 'bash'], default='dot', help="Output file type")
    parser.add_argument("--filter_path", help="Remove path prefix for script output")
    args = parser.parse_args()

    node_dir = os.environ["BART_TOOL_GRAPH"]
    graph = Graph(args.infile, node_dir)

    if args.type == 'dot':
        print(graph.get_dot())
    elif args.type == 'bash':
        print(graph.create_cmds(args.filter_path))
