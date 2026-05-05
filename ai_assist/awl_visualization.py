"""AWL workflow visualization using Dagre + D3.js flowchart"""

import html
import json
import webbrowser
from pathlib import Path
from typing import Any

from .awl_ast import (
    ASTNode,
    FailNode,
    GoalNode,
    IfNode,
    LoopNode,
    NotifyNode,
    ReturnNode,
    SetNode,
    TaskNode,
    WaitNode,
    WhileNode,
    WorkflowNode,
)
from .awl_parser import AWLParser
from .config import get_config_dir
from .introspection_tools import IntrospectionTools

NODE_STYLES = {
    "task": {"color": "#3498db", "bg": "#1a3a5c", "icon": "⚙", "label": "Task"},
    "if": {"color": "#e67e22", "bg": "#5c3a1a", "icon": "⑂", "label": "Condition"},
    "loop": {"color": "#2ecc71", "bg": "#1a5c3a", "icon": "↻", "label": "Loop"},
    "while": {"color": "#27ae60", "bg": "#1a4c2a", "icon": "↺", "label": "While"},
    "wait": {"color": "#95a5a6", "bg": "#3a3a3a", "icon": "⏳", "label": "Wait"},
    "notify": {"color": "#9b59b6", "bg": "#3a1a5c", "icon": "♪", "label": "Notify"},
    "set": {"color": "#1abc9c", "bg": "#1a4c4c", "icon": "←", "label": "Set"},
    "return": {"color": "#2ecc71", "bg": "#1a5c3a", "icon": "↵", "label": "Return"},
    "fail": {"color": "#e74c3c", "bg": "#5c1a1a", "icon": "✘", "label": "Fail"},
    "goal": {"color": "#f1c40f", "bg": "#5c4c1a", "icon": "⚑", "label": "Goal"},
    "merge": {"color": "#888888", "bg": "#333333", "icon": "", "label": "Merge"},
}


def discover_awl_scripts() -> list[Path]:
    """Discover .awl scripts from CWD, config dir, and installed skills."""
    seen: set[Path] = set()
    results: list[Path] = []

    sources = [
        Path.cwd(),
        get_config_dir(),
    ]

    skills_cache = get_config_dir() / "skills-cache"
    if skills_cache.is_dir():
        sources.append(skills_cache)

    for base in sources:
        if not base.is_dir():
            continue
        for awl_file in sorted(base.rglob("*.awl")):
            resolved = awl_file.resolve()
            if resolved not in seen:
                seen.add(resolved)
                results.append(awl_file)

    return results


def _count_tasks(nodes: list[ASTNode]) -> int:
    """Count total TaskNode instances in an AST."""
    count = 0
    for node in nodes:
        if isinstance(node, TaskNode):
            count += 1
        elif isinstance(node, IfNode):
            count += _count_tasks(node.then_body) + _count_tasks(node.else_body)
        elif isinstance(node, (LoopNode, WhileNode, GoalNode)):
            count += _count_tasks(node.body)
    return count


def _truncate(text: str, max_len: int = 120) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def _format_duration(seconds: int) -> str:
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


class _GraphBuilder:
    """Walks an AWL AST and builds a graph descriptor for dagre rendering."""

    def __init__(self) -> None:
        self._counter = 0
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.clusters: list[dict[str, Any]] = []
        self.back_edges: list[dict[str, Any]] = []

    def _next_id(self, prefix: str = "node") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def _add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        sublabel: str = "",
        style_key: str | None = None,
        parent_cluster: str | None = None,
        tooltip_data: dict[str, Any] | None = None,
        shape: str = "rect",
    ) -> None:
        self.nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "label": label,
                "sublabel": sublabel,
                "styleKey": style_key or node_type,
                "parentCluster": parent_cluster,
                "tooltipData": tooltip_data,
                "shape": shape,
            }
        )

    def _add_edge(
        self,
        source: str,
        target: str,
        labels: list[str] | None = None,
        style: str = "normal",
    ) -> None:
        self.edges.append({"source": source, "target": target, "labels": labels or [], "style": style})

    def _add_back_edge(self, source: str, target: str) -> None:
        self.back_edges.append({"source": source, "target": target, "labels": [], "style": "back"})

    def _add_cluster(self, cluster_id: str, label: str, cluster_type: str, parent: str | None = None) -> None:
        self.clusters.append({"id": cluster_id, "label": label, "type": cluster_type, "parentCluster": parent})

    def process_sequence(
        self, ast_nodes: list[ASTNode], parent_cluster_id: str | None
    ) -> tuple[str | None, str | None]:
        if not ast_nodes:
            return (None, None)

        prev_exit: str | None = None
        first_entry: str | None = None
        pending_labels: list[str] = []

        for node in ast_nodes:
            entry_id, exit_id, out_labels = self._process_node(node, parent_cluster_id)
            if prev_exit is not None:
                self._add_edge(prev_exit, entry_id, labels=pending_labels)
            if first_entry is None:
                first_entry = entry_id
            pending_labels = out_labels
            prev_exit = exit_id

        return (first_entry, prev_exit)

    def _process_node(self, node: ASTNode, pcid: str | None) -> tuple[str, str, list[str]]:
        dispatch: dict[type, str] = {
            TaskNode: "_process_task",
            SetNode: "_process_set",
            WaitNode: "_process_wait",
            NotifyNode: "_process_notify",
            ReturnNode: "_process_return",
            FailNode: "_process_fail",
            IfNode: "_process_if",
            LoopNode: "_process_loop",
            WhileNode: "_process_while",
            GoalNode: "_process_goal",
        }
        method_name = dispatch.get(type(node))
        if method_name:
            return getattr(self, method_name)(node, pcid)
        nid = self._next_id("unknown")
        self._add_node(nid, "task", label="???", parent_cluster=pcid)
        return (nid, nid, [])

    def _process_task(self, node: TaskNode, pcid: str | None) -> tuple[str, str, list[str]]:
        nid = self._next_id("task")
        tooltip = {
            "task_id": node.task_id,
            "goal": node.goal,
            "context": node.context,
            "constraints": node.constraints,
            "success": node.success,
            "hints": node.hints,
            "max_tool_calls": node.max_tool_calls,
            "max_time": node.max_time,
            "expose": node.expose,
        }
        self._add_node(
            nid,
            "task",
            label=f"task: {node.task_id}",
            sublabel=_truncate(node.goal, 60) if node.goal else "",
            parent_cluster=pcid,
            tooltip_data=tooltip,
        )
        return (nid, nid, list(node.expose))

    def _process_set(self, node: SetNode, pcid: str | None) -> tuple[str, str, list[str]]:
        nid = self._next_id("set")
        self._add_node(
            nid,
            "set",
            label=f"set {node.variable}",
            sublabel=_truncate(node.value, 60),
            parent_cluster=pcid,
        )
        return (nid, nid, [node.variable])

    def _process_wait(self, node: WaitNode, pcid: str | None) -> tuple[str, str, list[str]]:
        nid = self._next_id("wait")
        self._add_node(
            nid,
            "wait",
            label=f"wait {_format_duration(node.duration_seconds)}",
            parent_cluster=pcid,
        )
        return (nid, nid, [])

    def _process_notify(self, node: NotifyNode, pcid: str | None) -> tuple[str, str, list[str]]:
        nid = self._next_id("notify")
        self._add_node(
            nid,
            "notify",
            label="notify",
            sublabel=_truncate(node.message, 60),
            parent_cluster=pcid,
        )
        return (nid, nid, [])

    def _process_return(self, node: ReturnNode, pcid: str | None) -> tuple[str, str, list[str]]:
        nid = self._next_id("return")
        self._add_node(nid, "return", label=f"return {node.expression}", parent_cluster=pcid)
        return (nid, nid, [])

    def _process_fail(self, node: FailNode, pcid: str | None) -> tuple[str, str, list[str]]:
        nid = self._next_id("fail")
        self._add_node(
            nid,
            "fail",
            label="fail",
            sublabel=_truncate(node.message, 60),
            parent_cluster=pcid,
        )
        return (nid, nid, [])

    def _process_if(self, node: IfNode, pcid: str | None) -> tuple[str, str, list[str]]:
        dec_id = self._next_id("decision")
        self._add_node(
            dec_id,
            "if",
            label=f"if {_truncate(node.expression, 40)}",
            parent_cluster=pcid,
            shape="diamond",
        )
        merge_id = self._next_id("merge")
        self._add_node(merge_id, "merge", label="", parent_cluster=pcid, shape="circle")

        self._connect_branch(dec_id, merge_id, node.then_body, pcid, "then")
        if node.else_body:
            self._connect_branch(dec_id, merge_id, node.else_body, pcid, "else")
        else:
            self._add_edge(dec_id, merge_id, labels=["else"], style="else")

        return (dec_id, merge_id, [])

    def _connect_branch(
        self,
        dec_id: str,
        merge_id: str,
        body: list[ASTNode],
        pcid: str | None,
        branch_label: str,
    ) -> None:
        entry, exit_ = self.process_sequence(body, pcid)
        if entry and exit_:
            self._add_edge(dec_id, entry, labels=[branch_label], style=branch_label)
            self._add_edge(exit_, merge_id)
        else:
            self._add_edge(dec_id, merge_id, labels=[branch_label], style=branch_label)

    def _process_loop(self, node: LoopNode, pcid: str | None) -> tuple[str, str, list[str]]:
        cluster_id = self._next_id("cluster_loop")
        meta_parts = []
        if node.limit is not None:
            meta_parts.append(f"limit={node.limit}")
        if node.collect:
            collect_str = node.collect
            if node.collect_fields:
                collect_str += f"({', '.join(node.collect_fields)})"
            meta_parts.append(f"collect={collect_str}")
        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
        self._add_cluster(cluster_id, f"loop {node.collection} as {node.item_var}{meta}", "loop", pcid)

        body_entry, body_exit = self.process_sequence(node.body, cluster_id)
        self._add_loop_back_edge(body_entry, body_exit)

        out_labels = [node.collect] if node.collect else []
        entry = body_entry or self._next_id("loop_empty")
        return (entry, body_exit or entry, out_labels)

    def _process_while(self, node: WhileNode, pcid: str | None) -> tuple[str, str, list[str]]:
        cluster_id = self._next_id("cluster_while")
        meta = f" (max_iterations={node.max_iterations})" if node.max_iterations != 100 else ""
        self._add_cluster(cluster_id, f"while {node.expression}{meta}", "while", pcid)

        body_entry, body_exit = self.process_sequence(node.body, cluster_id)
        self._add_loop_back_edge(body_entry, body_exit)

        entry = body_entry or self._next_id("while_empty")
        return (entry, body_exit or entry, [])

    def _process_goal(self, node: GoalNode, pcid: str | None) -> tuple[str, str, list[str]]:
        cluster_id = self._next_id("cluster_goal")
        self._add_cluster(cluster_id, f"goal: {node.goal_id} (max_actions={node.max_actions})", "goal", pcid)
        body_entry, body_exit = self.process_sequence(node.body, cluster_id)
        entry = body_entry or self._next_id("goal_empty")
        return (entry, body_exit or entry, [])

    def _add_loop_back_edge(self, body_entry: str | None, body_exit: str | None) -> None:
        if body_entry and body_exit and body_entry != body_exit:
            self._add_back_edge(body_exit, body_entry)
        elif body_entry and body_exit:
            self._add_back_edge(body_entry, body_entry)

    def build(self, workflow: WorkflowNode) -> dict[str, Any]:
        self.process_sequence(workflow.body, None)
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "clusters": self.clusters,
            "backEdges": self.back_edges,
        }


def _build_graph(workflow: WorkflowNode) -> dict[str, Any]:
    """Walk the AST and build a graph descriptor for dagre rendering."""
    return _GraphBuilder().build(workflow)


_RENDER_JS = r"""
(function() {
  var g = new dagre.graphlib.Graph({compound: true})
    .setGraph({rankdir: 'TB', nodesep: 70, ranksep: 60, marginx: 30, marginy: 30, edgesep: 20})
    .setDefaultEdgeLabel(function() { return {}; });

  GRAPH_DATA.clusters.forEach(function(c) {
    g.setNode(c.id, {label: c.label, clusterLabelPos: 'top',
      paddingTop: 35, paddingBottom: 20, paddingLeft: 20, paddingRight: 20});
    if (c.parentCluster) g.setParent(c.id, c.parentCluster);
  });

  GRAPH_DATA.nodes.forEach(function(n) {
    var w = n.shape === 'diamond' ? 140 : n.shape === 'circle' ? 16 : Math.max(180, Math.min(280, n.label.length * 9 + 40));
    var h = n.shape === 'diamond' ? 70 : n.shape === 'circle' ? 16 : (n.sublabel ? 58 : 40);
    g.setNode(n.id, {label: n.label, width: w, height: h, _data: n});
    if (n.parentCluster) g.setParent(n.id, n.parentCluster);
  });

  GRAPH_DATA.edges.forEach(function(e) {
    g.setEdge(e.source, e.target, {label: e.labels.join(', '), _style: e.style, labeloffset: 10, minlen: 1});
  });

  dagre.layout(g);

  var svg = d3.select('#workflow-svg');
  var svgGroup = svg.append('g').attr('class', 'graph-root');

  // Arrow markers
  var defs = svg.append('defs');
  defs.append('marker').attr('id', 'arrow').attr('viewBox', '0 0 10 10')
    .attr('refX', 9).attr('refY', 5).attr('markerWidth', 7).attr('markerHeight', 7)
    .attr('orient', 'auto').append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', '#666');
  defs.append('marker').attr('id', 'arrow-back').attr('viewBox', '0 0 10 10')
    .attr('refX', 9).attr('refY', 5).attr('markerWidth', 7).attr('markerHeight', 7)
    .attr('orient', 'auto').append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', '#2ecc71');

  // Draw clusters
  GRAPH_DATA.clusters.forEach(function(c) {
    var cn = g.node(c.id);
    if (!cn) return;
    var st = NODE_STYLES_DATA[c.type] || NODE_STYLES_DATA['task'];
    var cg = svgGroup.append('g').attr('class', 'cluster');
    cg.append('rect')
      .attr('x', cn.x - cn.width/2).attr('y', cn.y - cn.height/2)
      .attr('width', cn.width).attr('height', cn.height)
      .attr('rx', 10).attr('fill', 'rgba(255,255,255,0.03)')
      .attr('stroke', st.color).attr('stroke-dasharray', '8,4').attr('stroke-width', 1.5);
    cg.append('text')
      .attr('x', cn.x - cn.width/2 + 12).attr('y', cn.y - cn.height/2 + 18)
      .attr('fill', st.color).attr('font-size', '12px').attr('font-weight', '600')
      .text(st.icon + ' ' + c.label);
  });

  // Draw edges
  var line = d3.line().x(function(d) { return d.x; }).y(function(d) { return d.y; }).curve(d3.curveBasis);
  g.edges().forEach(function(e) {
    var edge = g.edge(e);
    svgGroup.append('path')
      .attr('d', line(edge.points)).attr('fill', 'none')
      .attr('stroke', '#555').attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#arrow)');
    if (edge.label) {
      var mid = edge.points[Math.floor(edge.points.length / 2)];
      var bg = svgGroup.append('rect')
        .attr('rx', 3).attr('fill', '#1a1a2e').attr('stroke', '#333').attr('stroke-width', 0.5);
      var txt = svgGroup.append('text')
        .attr('x', mid.x + 8).attr('y', mid.y)
        .attr('fill', '#7fb3d8').attr('font-size', '11px').attr('font-weight', '500')
        .attr('dominant-baseline', 'middle').text(edge.label);
      var bb = txt.node().getBBox();
      bg.attr('x', bb.x - 3).attr('y', bb.y - 1).attr('width', bb.width + 6).attr('height', bb.height + 2);
    }
  });

  // Draw nodes
  GRAPH_DATA.nodes.forEach(function(n) {
    var gn = g.node(n.id);
    if (!gn) return;
    var st = NODE_STYLES_DATA[n.styleKey] || NODE_STYLES_DATA['task'];
    var ng = svgGroup.append('g').attr('class', 'graph-node')
      .attr('data-node-id', n.id)
      .attr('transform', 'translate(' + gn.x + ',' + gn.y + ')');

    if (n.shape === 'diamond') {
      var dw = gn.width/2, dh = gn.height/2;
      ng.append('polygon')
        .attr('points', '0,' + (-dh) + ' ' + dw + ',0 0,' + dh + ' ' + (-dw) + ',0')
        .attr('fill', st.bg).attr('stroke', st.color).attr('stroke-width', 2);
      ng.append('text').attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('fill', '#e0e0e0').attr('font-size', '11px').text(n.label);
    } else if (n.shape === 'circle') {
      ng.append('circle').attr('r', 8).attr('fill', '#444').attr('stroke', '#888').attr('stroke-width', 1);
    } else {
      ng.append('rect')
        .attr('x', -gn.width/2).attr('y', -gn.height/2)
        .attr('width', gn.width).attr('height', gn.height)
        .attr('rx', 8).attr('fill', st.bg).attr('stroke', st.color).attr('stroke-width', 2);
      var ty = n.sublabel ? -6 : 0;
      ng.append('text').attr('text-anchor', 'middle').attr('y', ty)
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#e0e0e0').attr('font-size', '13px').attr('font-weight', '500')
        .text(st.icon + ' ' + n.label);
      if (n.sublabel) {
        ng.append('text').attr('text-anchor', 'middle').attr('y', ty + 18)
          .attr('dominant-baseline', 'middle')
          .attr('fill', '#999').attr('font-size', '11px')
          .text(n.sublabel.length > 40 ? n.sublabel.substring(0, 40) + '...' : n.sublabel);
      }
    }

    if (n.tooltipData) {
      ng.style('cursor', 'pointer');
      ng.on('mouseenter', function(event) { showTooltip(event, n.tooltipData); })
        .on('mousemove', function(event) { moveTooltip(event); })
        .on('mouseleave', function() { hideTooltip(); });
    }
  });

  // Draw back-edges (loop arrows)
  GRAPH_DATA.backEdges.forEach(function(be) {
    var srcN = g.node(be.source);
    var tgtN = g.node(be.target);
    if (!srcN || !tgtN) return;
    var offset = 50;
    var maxX = Math.max(srcN.x + srcN.width/2, tgtN.x + tgtN.width/2) + offset;
    var path = 'M ' + (srcN.x + srcN.width/2) + ' ' + srcN.y
      + ' C ' + maxX + ' ' + srcN.y + ', ' + maxX + ' ' + tgtN.y + ', '
      + (tgtN.x + tgtN.width/2) + ' ' + tgtN.y;
    svgGroup.append('path').attr('d', path).attr('fill', 'none')
      .attr('stroke', '#2ecc71').attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,3').attr('marker-end', 'url(#arrow-back)');
  });

  // Fit SVG
  var bbox = svgGroup.node().getBBox();
  svg.attr('viewBox', (bbox.x - 30) + ' ' + (bbox.y - 30) + ' ' + (bbox.width + 60) + ' ' + (bbox.height + 60))
     .attr('width', '100%').attr('height', Math.max(400, bbox.height + 60));

  // Zoom + pan
  var zoom = d3.zoom().scaleExtent([0.3, 3]).on('zoom', function(event) {
    svgGroup.attr('transform', event.transform);
  });
  svg.call(zoom);

  // Tooltip
  var tooltip = document.getElementById('tooltip');
  function showTooltip(event, data) {
    var h = '<div class="tt-title">' + esc(data.task_id) + '</div>';
    if (data.goal) h += '<div class="tt-field"><strong>Goal:</strong> ' + esc(data.goal) + '</div>';
    if (data.context) h += '<div class="tt-field"><strong>Context:</strong> ' + esc(data.context) + '</div>';
    if (data.constraints) h += '<div class="tt-field"><strong>Constraints:</strong> ' + esc(data.constraints) + '</div>';
    if (data.success) h += '<div class="tt-field"><strong>Success:</strong> ' + esc(data.success) + '</div>';
    if (data.hints && data.hints.length) h += '<div class="tt-field"><strong>Hints:</strong> ' + data.hints.map(function(x) { return '<span class="tt-tag">' + esc(x) + '</span>'; }).join(' ') + '</div>';
    if (data.max_tool_calls != null) h += '<div class="tt-field"><strong>Max tool calls:</strong> ' + data.max_tool_calls + '</div>';
    if (data.max_time != null) h += '<div class="tt-field"><strong>Max time:</strong> ' + data.max_time + 's</div>';
    if (data.expose && data.expose.length) h += '<div class="tt-field tt-expose"><strong>Expose:</strong> ' + data.expose.join(', ') + '</div>';
    tooltip.innerHTML = h;
    tooltip.classList.remove('hidden');
    moveTooltip(event);
  }
  function moveTooltip(event) {
    var x = event.clientX + 15, y = event.clientY + 15;
    var r = tooltip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 10) x = event.clientX - r.width - 15;
    if (y + r.height > window.innerHeight - 10) y = event.clientY - r.height - 15;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }
  function hideTooltip() { tooltip.classList.add('hidden'); }
  function esc(s) { var d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
})();
"""


def generate_awl_html(workflow: WorkflowNode, title: str = "AWL Workflow") -> str:
    """Generate an interactive HTML visualization of an AWL workflow."""
    if not workflow.body:
        return (
            '<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
            "<title>AWL Workflow</title>\n"
            "<style>body{font-family:sans-serif;display:flex;align-items:center;"
            "justify-content:center;height:100vh;margin:0;background:#1a1a2e;"
            "color:#e0e0e0;}\n"
            ".msg{text-align:center;}.msg h1{font-size:2em;margin-bottom:0.5em;}"
            ".msg p{color:#888;}</style>\n"
            '</head><body><div class="msg"><h1>Empty Workflow</h1>'
            "<p>This AWL script has no directives.</p></div></body></html>"
        )

    task_count = _count_tasks(workflow.body)
    input_vars = IntrospectionTools._awl_input_variables(workflow)
    input_vars_str = ", ".join(sorted(input_vars)) if input_vars else "none"
    max_steps_str = str(workflow.max_steps) if workflow.max_steps else "default"

    graph_data = _build_graph(workflow)
    graph_json = json.dumps(graph_data, ensure_ascii=False)
    styles_json = json.dumps(NODE_STYLES, ensure_ascii=False)

    legend_items = []
    for key, style in NODE_STYLES.items():
        if key == "merge":
            continue
        legend_items.append(
            f'<span class="legend-item">'
            f'<span class="legend-dot" style="color:{style["color"]};">●</span>'
            f" {style['label']}</span>"
        )
    legend_html = " ".join(legend_items)

    escaped_title = html.escape(title)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{escaped_title} - AWL Visualization</title>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/d3@7/dist/d3.min.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }}
#header {{ padding: 12px 24px; background: #16213e; border-bottom: 1px solid #0f3460; display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }}
#header h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
#header .stat {{ font-size: 13px; color: #888; }}
#header .stat strong {{ color: #b0b0b0; }}
#legend {{ padding: 8px 24px; background: #0f3460; font-size: 13px; display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
.legend-item {{ display: inline-flex; align-items: center; gap: 4px; }}
.legend-dot {{ font-size: 14px; }}
#graph-container {{ width: 100%; overflow: auto; padding: 16px; }}
#workflow-svg {{ display: block; margin: 0 auto; }}
.tooltip {{ position: fixed; background: #16213e; border: 1px solid #3498db; border-radius: 8px; padding: 14px 18px; max-width: 520px; z-index: 1000; font-size: 13px; line-height: 1.6; box-shadow: 0 4px 16px rgba(0,0,0,0.6); pointer-events: none; }}
.tooltip.hidden {{ display: none; }}
.tt-title {{ font-size: 15px; font-weight: 600; color: #3498db; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #2a3a5e; }}
.tt-field {{ margin-bottom: 5px; }}
.tt-field strong {{ color: #b0b0b0; }}
.tt-tag {{ font-size: 11px; background: #2a2a4a; color: #aaa; padding: 1px 6px; border-radius: 3px; margin-left: 2px; }}
.tt-expose {{ color: #7fb3d8; }}
</style>
</head>
<body>
<div id="header">
  <h1>{escaped_title}</h1>
  <span class="stat"><strong>{task_count}</strong> tasks</span>
  <span class="stat">Inputs: <strong>{html.escape(input_vars_str)}</strong></span>
  <span class="stat">Max steps: <strong>{html.escape(max_steps_str)}</strong></span>
</div>
<div id="legend">{legend_html}</div>
<div id="graph-container">
  <svg id="workflow-svg"></svg>
</div>
<div id="tooltip" class="tooltip hidden"></div>
<script>
var GRAPH_DATA = {graph_json};
var NODE_STYLES_DATA = {styles_json};
{_RENDER_JS}
</script>
</body>
</html>"""


def open_awl_visualization(script_path: str) -> str:
    """Parse an AWL script and open its visualization in the browser."""
    path = Path(script_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    source = path.read_text()
    workflow = AWLParser(source).parse()
    title = path.name

    html_content = generate_awl_html(workflow, title)
    filepath = get_config_dir() / "awl_visualization.html"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(html_content)
    webbrowser.open(f"file://{filepath}")
    return str(filepath)
