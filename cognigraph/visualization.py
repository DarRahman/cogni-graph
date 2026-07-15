# cognigraph/visualization.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""HTML visualization generator for CogniGraph memory graphs."""

import json
import logging
from typing import Any, Dict, List
from cognigraph.models import Entity, Relationship
from cognigraph.graph_store import GraphStore

logger = logging.getLogger("cognigraph.visualization")


def generate_visual_html(graph_store: GraphStore) -> str:
    """Generates a standalone HTML page with an interactive vis.js network visualization.

    Args:
        graph_store: The graph store containing the entities and relationships.

    Returns:
        A string containing the full HTML document.
    """
    logger.info("Generating interactive HTML visualization from graph store")
    
    entities = graph_store.get_all_entities()
    relationships = graph_store.get_all_relationships()

    nodes_data: List[Dict[str, Any]] = []
    for ent in entities:
        node_dict = ent.model_dump(mode="json")
        node_dict["degree"] = graph_store.get_degree(ent.id)
        nodes_data.append(node_dict)

    edges_data: List[Dict[str, Any]] = []
    for rel in relationships:
        edges_data.append(rel.model_dump(mode="json"))

    # Standalone HTML template with vis-network
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>CogniGraph Memory Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        #container {
            display: flex;
            width: 100%;
            height: 100%;
        }
        #mynetwork {
            flex-grow: 1;
            height: 100%;
            background-color: #ffffff;
            border-right: 1px solid #dee2e6;
        }
        #sidebar {
            width: 350px;
            padding: 20px;
            background-color: #f1f3f5;
            box-shadow: -2px 0 5px rgba(0,0,0,0.05);
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        h2 {
            margin-top: 0;
            color: #343a40;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
        }
        .section {
            margin-bottom: 20px;
        }
        .label {
            font-weight: bold;
            color: #495057;
        }
        .value {
            margin-bottom: 10px;
            color: #212529;
            word-break: break-word;
        }
        .property-tag {
            display: inline-block;
            background-color: #e9ecef;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        #details {
            flex-grow: 1;
        }
        .placeholder {
            color: #868e96;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="mynetwork"></div>
        <div id="sidebar">
            <h2>Memory Inspector</h2>
            <div id="details">
                <p class="placeholder">Click on a node or edge to inspect details.</p>
            </div>
        </div>
    </div>

    <script type="text/javascript">
        // Inject nodes and edges
        const nodesData = {NODES_JSON};
        const edgesData = {EDGES_JSON};

        // Color palette for entity types
        const colors = {
            'person': '#4dabf7',
            'organization': '#37b24d',
            'location': '#f76707',
            'technology': '#7048e8',
            'concept': '#ae3ec9',
            'project': '#1098ad',
            'default': '#748ffc'
        };

        const nodes = new vis.DataSet(nodesData.map(node => {
            const typeLower = (node.type || '').toLowerCase();
            const color = colors[typeLower] || colors['default'];
            return {
                id: node.id,
                label: node.name,
                title: `${node.name} (${node.type})`,
                color: {
                    background: color,
                    border: color,
                    highlight: {
                        background: color,
                        border: '#212529'
                    }
                },
                font: { color: '#ffffff' },
                shape: 'dot',
                size: 15 + (node.degree || 0) * 2,
                extendedData: node
            };
        }));

        const edges = new vis.DataSet(edgesData.map(edge => {
            return {
                id: `${edge.source}-${edge.target}-${edge.type}`,
                from: edge.source,
                to: edge.target,
                label: edge.type,
                title: `${edge.source} -[${edge.type}]-> ${edge.target}`,
                arrows: 'to',
                width: Math.max(1, edge.weight * 2),
                color: { color: '#adb5bd', highlight: '#495057' },
                font: { align: 'top', size: 10 },
                extendedData: edge
            };
        }));

        const container = document.getElementById('mynetwork');
        const data = { nodes: nodes, edges: edges };
        const options = {
            physics: {
                stabilization: true,
                barnesHut: {
                    gravitationalConstant: -2000,
                    centralGravity: 0.3,
                    springLength: 95,
                    springConstant: 0.04,
                    damping: 0.09,
                    avoidOverlap: 0.1
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200
            }
        };

        const network = new vis.Network(container, data, options);

        network.on("click", function (params) {
            const detailsDiv = document.getElementById('details');
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId).extendedData;
                
                let html = `
                    <div class="section">
                        <div class="label">Name</div>
                        <div class="value">${node.name}</div>
                    </div>
                    <div class="section">
                        <div class="label">Type</div>
                        <div class="value"><span class="property-tag">${node.type}</span></div>
                    </div>
                `;
                
                if (node.description) {
                    html += `
                        <div class="section">
                            <div class="label">Description</div>
                            <div class="value">${node.description}</div>
                        </div>
                    `;
                }
                
                if (node.properties && Object.keys(node.properties).length > 0) {
                    html += '<div class="section"><div class="label">Properties</div><div class="value">';
                    for (const [k, v] of Object.entries(node.properties)) {
                        html += `<span class="property-tag"><strong>${k}:</strong> ${JSON.stringify(v)}</span>`;
                    }
                    html += '</div></div>';
                }
                
                html += `
                    <div class="section">
                        <div class="label">Timestamps</div>
                        <div class="value" style="font-size: 0.85em; color: #6c757d;">
                            Created: ${node.created_at}<br>
                            Updated: ${node.updated_at}
                        </div>
                    </div>
                `;
                
                detailsDiv.innerHTML = html;
            } else if (params.edges.length > 0) {
                const edgeId = params.edges[0];
                const edge = edges.get(edgeId).extendedData;
                
                let html = `
                    <div class="section">
                        <div class="label">Relationship</div>
                        <div class="value">${edge.source} &rarr; ${edge.target}</div>
                    </div>
                    <div class="section">
                        <div class="label">Type</div>
                        <div class="value"><span class="property-tag">${edge.type}</span></div>
                    </div>
                    <div class="section">
                        <div class="label">Weight</div>
                        <div class="value">${edge.weight.toFixed(4)}</div>
                    </div>
                `;
                
                if (edge.description) {
                    html += `
                        <div class="section">
                            <div class="label">Context</div>
                            <div class="value">${edge.description}</div>
                        </div>
                    `;
                }
                
                if (edge.properties && Object.keys(edge.properties).length > 0) {
                    html += '<div class="section"><div class="label">Properties</div><div class="value">';
                    for (const [k, v] of Object.entries(edge.properties)) {
                        html += `<span class="property-tag"><strong>${k}:</strong> ${JSON.stringify(v)}</span>`;
                    }
                    html += '</div></div>';
                }
                
                html += `
                    <div class="section">
                        <div class="label">Timestamps</div>
                        <div class="value" style="font-size: 0.85em; color: #6c757d;">
                            Created: ${edge.created_at}<br>
                            Updated: ${edge.updated_at}
                        </div>
                    </div>
                `;
                
                detailsDiv.innerHTML = html;
            } else {
                detailsDiv.innerHTML = '<p class="placeholder">Click on a node or edge to inspect details.</p>';
            }
        });
    </script>
</body>
</html>"""

    # Replace placeholders with JSON strings
    html_content = html_template.replace("{NODES_JSON}", json.dumps(nodes_data))
    html_content = html_content.replace("{EDGES_JSON}", json.dumps(edges_data))
    
    return html_content
