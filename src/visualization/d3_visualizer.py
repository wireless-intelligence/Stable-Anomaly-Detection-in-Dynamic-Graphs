import os
import json
import numpy as np

D3_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SAG Graph Anomaly Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .chart-box {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
        }}
        .chart-title {{
            font-size: 14px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 10px;
            color: #333;
        }}
        svg {{
            display: block;
            margin: 0 auto;
        }}
        .node {{
            stroke: #fff;
            stroke-width: 1.5px;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        .legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.9);
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }}
        .main-legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .main-legend-item {{
            display: flex;
            align-items: center;
            font-size: 14px;
        }}
        .main-legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 8px;
            border: 2px solid #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
    </style>
</head>
<body>
    <h1>SAG - Graph Anomaly Detection Visualization</h1>
    <div class="main-legend">
        <div class="main-legend-item">
            <div class="main-legend-color" style="background: #3498DB;"></div>
            <span>Normal Nodes</span>
        </div>
        <div class="main-legend-item">
            <div class="main-legend-color" style="background: #E74C3C;"></div>
            <span>Predicted Anomaly</span>
        </div>
        <div class="main-legend-item">
            <div class="main-legend-color" style="background: #2ECC71;"></div>
            <span>Ground Truth Anomaly</span>
        </div>
        <div class="main-legend-item">
            <div class="main-legend-color" style="background: #9B59B6;"></div>
            <span>True Positive</span>
        </div>
    </div>
    <div class="container" id="charts"></div>
    
    <script>
        const data = {graph_data};
        
        const width = 280;
        const height = 250;
        
        function createChart(container, subset, title) {{
            const box = container.append("div")
                .attr("class", "chart-box");
            
            box.append("div")
                .attr("class", "chart-title")
                .text(title);
            
            const svg = box.append("svg")
                .attr("width", width)
                .attr("height", height);
            
            const g = svg.append("g");
            
            const simulation = d3.forceSimulation(subset.nodes)
                .force("link", d3.forceLink(subset.links).id(d => d.id).distance(30))
                .force("charge", d3.forceManyBody().strength(-50))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(8));
            
            const link = g.append("g")
                .selectAll("line")
                .data(subset.links)
                .join("line")
                .attr("class", "link")
                .attr("stroke-width", 1);
            
            const node = g.append("g")
                .selectAll("circle")
                .data(subset.nodes)
                .join("circle")
                .attr("class", "node")
                .attr("r", d => 4 + d.score * 4)
                .attr("fill", d => d.color);
            
            node.append("title")
                .text(d => `Node ${{d.id}}\\nScore: ${{d.score.toFixed(3)}}\\nLabel: ${{d.label}}`);
            
            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node
                    .attr("cx", d => Math.max(10, Math.min(width - 10, d.x)))
                    .attr("cy", d => Math.max(10, Math.min(height - 10, d.y)));
            }});
        }}
        
        const container = d3.select("#charts");
        
        data.subsets.forEach((subset, i) => {{
            createChart(container, subset, subset.title);
        }});
    </script>
</body>
</html>'''


class D3Visualizer:
    def __init__(self, adj, scores, labels, num_samples=16, sample_size=30, seed=42):
        self.adj = adj
        self.scores = np.array(scores)
        self.labels = np.array(labels)
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.seed = seed
        
        np.random.seed(seed)
        
        if self.scores.max() > self.scores.min():
            self.scores_norm = (self.scores - self.scores.min()) / (self.scores.max() - self.scores.min())
        else:
            self.scores_norm = self.scores
    
    def _get_node_color(self, node_id, threshold=0.5):
        is_pred_anomaly = self.scores_norm[node_id] >= threshold
        is_true_anomaly = self.labels[node_id] == 1
        
        if is_pred_anomaly and is_true_anomaly:
            return '#9B59B6'
        elif is_true_anomaly:
            return '#2ECC71'
        elif is_pred_anomaly:
            return '#E74C3C'
        else:
            return '#3498DB'
    
    def _sample_subgraph(self, center_node, max_nodes=30):
        from scipy.sparse import csr_matrix
        
        if not hasattr(self, '_adj_csr'):
            self._adj_csr = csr_matrix(self.adj)
        
        visited = {center_node}
        queue = [center_node]
        
        while len(visited) < max_nodes and queue:
            current = queue.pop(0)
            neighbors = self._adj_csr[current].nonzero()[1]
            
            for neighbor in neighbors:
                if neighbor not in visited and len(visited) < max_nodes:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return list(visited)
    
    def _create_subset_data(self, node_ids, title, threshold=0.5):
        node_set = set(node_ids)
        id_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}
        
        nodes = []
        for node_id in node_ids:
            nodes.append({
                'id': id_map[node_id],
                'original_id': int(node_id),
                'score': float(self.scores_norm[node_id]),
                'label': int(self.labels[node_id]),
                'color': self._get_node_color(node_id, threshold)
            })
        
        links = []
        adj_coo = self.adj.tocoo()
        for i, j in zip(adj_coo.row, adj_coo.col):
            if i in node_set and j in node_set and i < j:
                links.append({
                    'source': id_map[i],
                    'target': id_map[j]
                })
        
        return {
            'title': title,
            'nodes': nodes,
            'links': links
        }
    
    def generate_visualization_data(self, threshold=0.5):
        subsets = []
        
        normal_indices = np.where(self.labels == 0)[0]
        anomaly_indices = np.where(self.labels == 1)[0]
        
        pred_anomaly_indices = np.where(self.scores_norm >= threshold)[0]
        true_positive_indices = np.intersect1d(pred_anomaly_indices, anomaly_indices)
        false_positive_indices = np.intersect1d(pred_anomaly_indices, normal_indices)
        false_negative_indices = np.setdiff1d(anomaly_indices, pred_anomaly_indices)
        
        num_each = self.num_samples // 4
        
        if len(normal_indices) > 0:
            for i in range(num_each):
                center = np.random.choice(normal_indices)
                nodes = self._sample_subgraph(center, self.sample_size)
                subsets.append(self._create_subset_data(nodes, f'Normal Region {i+1}', threshold))
        
        if len(true_positive_indices) > 0:
            for i in range(min(num_each, len(true_positive_indices))):
                center = true_positive_indices[i % len(true_positive_indices)]
                nodes = self._sample_subgraph(center, self.sample_size)
                subsets.append(self._create_subset_data(nodes, f'True Positive {i+1}', threshold))
        
        if len(anomaly_indices) > 0:
            for i in range(num_each):
                center = anomaly_indices[i % len(anomaly_indices)]
                nodes = self._sample_subgraph(center, self.sample_size)
                subsets.append(self._create_subset_data(nodes, f'Ground Truth {i+1}', threshold))
        
        if len(pred_anomaly_indices) > 0:
            for i in range(num_each):
                center = pred_anomaly_indices[i % len(pred_anomaly_indices)]
                nodes = self._sample_subgraph(center, self.sample_size)
                subsets.append(self._create_subset_data(nodes, f'Predicted {i+1}', threshold))
        
        while len(subsets) < self.num_samples:
            center = np.random.randint(len(self.labels))
            nodes = self._sample_subgraph(center, self.sample_size)
            subsets.append(self._create_subset_data(nodes, f'Sample {len(subsets)+1}', threshold))
        
        return {'subsets': subsets[:self.num_samples]}
    
    def generate_html(self, output_path, threshold=0.5):
        data = self.generate_visualization_data(threshold)
        
        html_content = D3_TEMPLATE.format(graph_data=json.dumps(data))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path


def create_d3_visualization(adj, scores, labels, output_path, 
                           num_samples=16, sample_size=30, threshold=0.5, seed=42):
    visualizer = D3Visualizer(adj, scores, labels, num_samples, sample_size, seed)
    return visualizer.generate_html(output_path, threshold)
