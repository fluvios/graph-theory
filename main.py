import mplleaflet
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from osm2nx import read_osm, haversine

from postman_problems.tests.utils import create_mock_csv_from_dataframe
from postman_problems.solver import cpp
from postman_problems.stats import calculate_postman_solution_stats

def contract_edges(graph, edge_weight='weight'):
    """
    Given a graph, contract edges into a list of contracted edges.  Nodes with degree 2 are collapsed into an edge
    stretching from a dead-end node (degree 1) or intersection (degree >= 3) to another like node.

    Args:
        graph (networkx graph):
        edge_weight (str): edge weight attribute to us for shortest path calculations

    Returns:
        List of tuples representing contracted edges
    """

    keep_nodes = [n for n in graph.nodes() if graph.degree(n) != 2]
    contracted_edges = []

    for n in keep_nodes:
        for nn in nx.neighbors(graph, n):

            nn_hood = set(nx.neighbors(graph, nn)) - {n}
            path = [n, nn]

            if len(nn_hood) == 1:
                while len(nn_hood) == 1:
                    nnn = list(nn_hood)[0]
                    nn_hood = set(nx.neighbors(graph, nnn)) - {path[-1]}
                    path += [nnn]

            full_edges = list(zip(path[:-1], path[1:]))  # granular edges between keep_nodes
            spl = sum([graph[e[0]][e[1]][edge_weight] for e in full_edges])  # distance

            # only keep if path is unique.  Parallel/Multi edges allowed, but not those that are completely redundant.
            if (not contracted_edges) | ([set(path)] not in [[set(p[3])] for p in contracted_edges]):
                contracted_edges.append(tuple(sorted([n, path[-1]])) + (spl,) + (path,))

    return contracted_edges

# load OSM to a directed NX
g_d = read_osm('datasets/sleepinggiant.osm')

# create an undirected graph
g = g_d.to_undirected()

g.add_edge('2318082790', '2318082832', id='white_horseshoe')

for e in g.edges(data=True):
    e[2]['distance'] = haversine(g.node[e[0]]['lon'],
                                 g.node[e[0]]['lat'],
                                 g.node[e[1]]['lon'],
                                 g.node[e[1]]['lat'])

g_t = g.copy()

for e in g.edges(data=True):

    # remove non trails
    name = e[2]['name'] if 'name' in e[2] else ''
    if ('Trail' not in name.split()) or (name is None):
        g_t.remove_edge(e[0], e[1])

    # remove non Sleeping Giant trails
    elif name in [
        'Farmington Canal Linear Trail',
        'Farmington Canal Heritage Trail',
        'Montowese Trail',
        '(white blazes)']:
        g_t.remove_edge(e[0], e[1])


edge_ids_to_add = [
    '223082783',
    '223077827',
    '40636272',
    '223082785',
    '222868698',
    '223083721',
    '222947116',
    '222711152',
    '222711155',
    '222860964',
    '223083718',
    '222867540',
    'white_horseshoe'
]

edge_ids_to_remove = [
    '17220599'
]

for e in g.edges(data=True):
    way_id = e[2].get('id').split('-')[0]
    if way_id in edge_ids_to_add:
        g_t.add_edge(e[0], e[1], **e[2])
        g_t.add_node(e[0], lat=g.node[e[0]]['lat'], lon=g.node[e[0]]['lon'])
        g_t.add_node(e[1], lat=g.node[e[1]]['lat'], lon=g.node[e[1]]['lon'])
    if way_id in edge_ids_to_remove:
        if g_t.has_edge(e[0], e[1]):
            g_t.remove_edge(e[0], e[1])

for n in nx.isolates(g_t.copy()):
    g_t.remove_node(n)

name2color = {
    'Green Trail': 'green',
    'Quinnipiac Trail': 'blue',
    'Tower Trail': 'black',
    'Yellow Trail': 'yellow',
    'Red Square Trail': 'red',
    'White/Blue Trail Link': 'lightblue',
    'Orange Trail': 'orange',
    'Mount Carmel Avenue': 'black',
    'Violet Trail': 'violet',
    'blue Trail': 'blue',
    'Red Triangle Trail': 'red',
    'Blue Trail': 'blue',
    'Blue/Violet Trail Link': 'purple',
    'Red Circle Trail': 'red',
    'White Trail': 'gray',
    'Red Diamond Trail': 'red',
    'Yellow/Green Trail Link': 'yellowgreen',
    'Nature Trail': 'forestgreen',
    'Red Hexagon Trail': 'red',
    None: 'black'
}

# print('{:0.2f} miles of required trail.'.format(sum([e[2]['distance']/1609.34 for e in g_t.edges(data=True)])))
#
# print('Number of edges in trail graph: {}'.format(len(g_t.edges())))

# intialize contracted graph
g_tc = nx.MultiGraph()

# add contracted edges to graph
for ce in contract_edges(g_t, 'distance'):
    start_node, end_node, distance, path = ce

    contracted_edge = {
        'start_node': start_node,
        'end_node': end_node,
        'distance': distance,
        'name': g[path[0]][path[1]].get('name'),
        'required': 1,
        'path': path
    }

    g_tc.add_edge(start_node, end_node, **contracted_edge)
    g_tc.node[start_node]['lat'] = g.node[start_node]['lat']
    g_tc.node[start_node]['lon'] = g.node[start_node]['lon']
    g_tc.node[end_node]['lat'] = g.node[end_node]['lat']
    g_tc.node[end_node]['lon'] = g.node[end_node]['lon']

# print('Number of edges in contracted trail graph: {}'.format(len(g_tc.edges())))

# create list with edge attributes and "from" & "to" nodes
tmp = []
for e in g_tc.edges(data=True):
    tmpi = e[2].copy()  # so we don't mess w original graph
    tmpi['start_node'] = e[0]
    tmpi['end_node'] = e[1]
    tmp.append(tmpi)

# create dataframe with node1 and node2 in order
eldf = pd.DataFrame(tmp)
eldf = eldf[['start_node', 'end_node'] + list(set(eldf.columns) - {'start_node', 'end_node'})]

# create edgelist mock CSV
elfn = create_mock_csv_from_dataframe(eldf)

circuit, graph = cpp(elfn, start_node='735393342')

print("----------- CPP Solutions -----------")
# print solution route
for e in circuit:
    print(e)

print("----------- CPP Summary -----------")
# print solution summary stats
for k, v in calculate_postman_solution_stats(circuit).items():
    print(k, v)

# convert 'path' from string back to list.  Caused by `create_mock_csv_from_dataframe`
for e in circuit:
    if type(e[3]['path']) == str:
        exec('e[3]["path"]=' + e[3]["path"])

# Create graph directly from circuit and original graph w lat/lon (g)
color_seq = [None, 'black', 'blue', 'orange', 'yellow']
grppviz = nx.MultiGraph()

for e in circuit:
    for n1, n2 in zip(e[3]['path'][:-1], e[3]['path'][1:]):
        if grppviz.has_edge(n1, n2):
            grppviz[n1][n2][0]['linewidth'] += 2
            grppviz[n1][n2][0]['cnt'] += 1
        else:
            grppviz.add_edge(n1, n2, linewidth=2.5)
            grppviz[n1][n2][0]['color_st'] = 'black' if g_t.has_edge(n1, n2) else 'red'
            grppviz[n1][n2][0]['cnt'] = 1
            grppviz.add_node(n1, lat=g.node[n1]['lat'], lon=g.node[n1]['lon'])
            grppviz.add_node(n2, lat=g.node[n2]['lat'], lon=g.node[n2]['lon'])

for e in grppviz.edges(data=True):
    e[2]['color_cnt'] = color_seq[1] if 'cnt' not in e[2] else color_seq[e[2]['cnt']]

fig, ax = plt.subplots(figsize=(1,10))

pos = {k: (grppviz.node[k]['lon'], grppviz.node[k]['lat']) for k in grppviz.nodes()}
e_width = [e[2]['linewidth'] for e in grppviz.edges(data=True)]
e_color = [e[2]['color_cnt'] for e in grppviz.edges(data=True)]
nx.draw_networkx_edges(grppviz, pos, width=e_width, edge_color=e_color, alpha=0.7)

mplleaflet.save_html(fig, 'maps/solutions.html', tiles='cartodb_positron')