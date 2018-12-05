import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import networkx as nx

class Plotter:
    def __init__(self, G):
        ""
        self.G = G

        self.pos = nx.get_node_attributes(self.G, 'pos')

        self.dmin = 1
        self.ncenter = 0

    def plot(self):

        for n in self.pos:
            x, y = self.pos[n]
            d=(x - 0.5) ** 2 + (y - 0.5) ** 2
            if d < self.dmin:
                self.ncenter = n
                self.dmin = d

        p = nx.single_source_shortest_path_length(self.G, self.ncenter)

        edge_trace = go.Scatter(
            x = [],
            y = [],
            line = dict(width = 0.5,color = '#888'),
            hoverinfo='none',
            mode='lines')

        for edge in self.G.edges():
            x0, y0 = self.G.node[edge[0]]['pos']
            x1, y1 = self.G.node[edge[1]]['pos']
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        node_trace = go.Scatter(
            x = [],
            y = [],
            text = [],
            mode = 'markers',
            hoverinfo = 'text',
            marker = dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar = dict(
                    thickness = 15,
                    title = 'Node Connections',
                    xanchor = 'left',
                    titleside = 'right'
                ),
            line = dict(width=2)))

        for node in self.G.nodes():
            x, y = self.G.node[node]['pos']
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])

        for node, adjacencies in enumerate(self.G.adjacency()):
            node_trace['marker']['color'] += tuple([len(adjacencies[1])])
            # node_info = '# of connections: ' + str(len(adjacencies[1]))
            node_info = 'Node: ' + str(node)
            node_trace['text'] += tuple([node_info])

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Community detection',
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        plotly.offline.plot(fig, filename='networkx.html')
