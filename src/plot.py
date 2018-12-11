import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import networkx as nx


class Plotter:
    def __init__(self, G, clusters):
        self.G = G
        self.clusters = clusters
        self.pos = nx.get_node_attributes(self.G, 'pos')
        self.color_dict = self.assign_color(clusters)
        self.dmin = 1
        self.ncenter = 0

    def assign_color(self, clusters):
        """Assigns a different color to each cluster given in clusters

        """
        color_dict = {}
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8',
                  '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                  '#bcf60c', '#fabebe', '#008080', '#e6beff',
                  '#9a6324', '#fffac8', '#800000', '#aaffc3',
                  '#808000', '#ffe8b1', '#000075', '#808080',
                  '#7ab301', '#000000', "#ff0000", "#9b59b6",
                  "#3498db", "#2ecc71", "#0000ff", "#FFE979",
                  "#58BABA", "#772F0F", '#f00', '#f00',
                  '#f00', '#f00', '#f00', '#f00', '#f00',
                  '#f00', '#f00', '#f00', '#f00', '#f00',
                  '#f00', '#f00', '#f00', '#f00', '#f00']
        for i, (lead, cluster) in enumerate(clusters.items()):
            color_dict[lead] = colors[i]
            for cluster_member in cluster:
                color_dict[cluster_member] = colors[i]

        return color_dict

    def plot(self):
        """Generates a scatter plot (nodes) and edge trace and puts them into
        a Figure that is plotted with plotly.

        """
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.8, color='#999'),
            hoverinfo='none',
            mode='lines')

        for edge in self.G.edges():
            x0, y0 = self.G.node[edge[0]]['pos']
            x1, y1 = self.G.node[edge[1]]['pos']
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=[],
                size=15,
                # line=dict(width=2)))
                line={'color': [],
                      'width': 2}))

        for node in self.G.nodes():
            x, y = self.G.node[node]['pos']
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_info = 'Node: ' + str(node)
            node_trace['text'] += tuple([node_info])
            if node in self.color_dict:
                node_trace['marker']['color'] += tuple([self.color_dict[node]])
                node_trace['marker']['line']['color'] += \
                    tuple([self.color_dict[node]])
            else:
                node_trace['marker']['color'] += tuple(['#000'])
                node_trace['marker']['line']['color'] += tuple(['#000'])

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Graph community detection',
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        plotly.offline.plot(fig, filename='networkx.html')
