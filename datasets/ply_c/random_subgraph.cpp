#pragma once
#include <boost/config.hpp>
#include <iostream>
#include <vector>
#include <boost/graph/adjacency_list.hpp>
#include <queue>

using namespace std;
using namespace boost;

namespace subgraph {

typedef adjacency_list <vecS, vecS, undirectedS> Graph;

typedef typename  boost::graph_traits< Graph >::vertex_descriptor VertexDescriptor;

typedef typename boost::property_map< Graph, boost::vertex_index_t>::type VertexIndexMap;

typedef typename graph_traits < Graph >::adjacency_iterator Adjacency_iterator;


void random_subgraph(const int n_ver, const int n_edg, const uint32_t * Eu, const uint32_t * Ev, int subgraph_size
                     , uint8_t * selected_edges,  uint8_t * selected_vertices)

{   //C-style interface

        if (n_ver < subgraph_size)
        {
            for (uint32_t i_edg = 0; i_edg < n_edg; i_edg++)
            {
                selected_edges[i_edg] = 1;
            }
            for (uint32_t i_ver = 0; i_ver < n_ver; i_ver++)
            {
                selected_vertices[n_ver] = 1;
            }
            return;
        }

        Graph G(n_ver);

        VertexIndexMap vertex_index_map = get(boost::vertex_index, G);
        VertexDescriptor ver_current;
        Adjacency_iterator ite_ver_adj,ite_ver_adj_end;
        int node_seen = 0, seed_index;
        queue<VertexDescriptor> ver_queue;

        for (uint32_t i_edg = 0; i_edg < n_edg; i_edg++)
        {   //building graph
            add_edge(vertex(Eu[i_edg],G), vertex(Ev[i_edg],G), G);
        }

        while(node_seen < subgraph_size)
        {
            //add seed vertex
            seed_index = rand() % n_ver;
            if (selected_vertices[seed_index])
            {
                continue;
            }
            ver_queue.push(vertex(seed_index,G));
            selected_vertices[vertex_index_map(ver_queue.front())] = 1;
            node_seen = node_seen + 1;

            while(!ver_queue.empty())
            {
                //pop the top of the queue and mark it as seen
                ver_current = ver_queue.front();
                ver_queue.pop();

                //add the neighbors of that vertex
                for (tie(ite_ver_adj, ite_ver_adj_end) = adjacent_vertices(ver_current, G); ite_ver_adj != ite_ver_adj_end; ite_ver_adj++)
                {
                    int i_ver_adj = vertex_index_map(*ite_ver_adj);
                    if ((selected_vertices[i_ver_adj]==0) && (node_seen <= subgraph_size))
                    {//vertex not already seen
                        node_seen = node_seen + 1;
                        selected_vertices[i_ver_adj] = 1;
                        ver_queue.push(*ite_ver_adj);
                    }

                    if (node_seen >= subgraph_size)
                    {//enough vertices
                        break;
                    }

                }
            }
        }

        for (int i_edg = 0; i_edg < n_edg; i_edg++)
        {   //add edges between selected vertices
            selected_edges[i_edg] = selected_vertices[vertex_index_map(vertex(Eu[i_edg],G))]
                                  * selected_vertices[vertex_index_map(vertex(Ev[i_edg],G))];
        }
    return;
}
}
