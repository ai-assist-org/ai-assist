"""Tests for AWL workflow visualization"""

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_assist.awl_ast import (
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
from ai_assist.awl_visualization import (
    NODE_STYLES,
    _build_graph,
    discover_awl_scripts,
    generate_awl_html,
    open_awl_visualization,
)


def _extract_graph_data(html_str: str) -> dict:
    """Extract the GRAPH_DATA JSON from the generated HTML."""
    match = re.search(r"var GRAPH_DATA = ({.*?});\s*var NODE_STYLES_DATA", html_str, re.DOTALL)
    assert match, "GRAPH_DATA not found in HTML"
    return json.loads(match.group(1))


@pytest.fixture
def simple_workflow():
    return WorkflowNode(
        body=[
            TaskNode(
                task_id="check_jobs",
                goal="Search for DCI jobs",
                hints=["no-history", "no-kg"],
                expose=["running_count"],
            ),
        ]
    )


@pytest.fixture
def complex_workflow():
    return WorkflowNode(
        max_steps=50,
        body=[
            SetNode(variable="prefix", value="Processing"),
            TaskNode(
                task_id="find_failures",
                goal="Search for failing jobs in the last 24 hours",
                hints=["no-history"],
                expose=["failed_jobs"],
                max_tool_calls=50,
            ),
            IfNode(
                expression="len(failed_jobs) > 0",
                then_body=[
                    LoopNode(
                        collection="failed_jobs",
                        item_var="job",
                        limit=5,
                        collect="analyses",
                        body=[
                            TaskNode(
                                task_id="analyze_failure",
                                goal="Classify the failure based on ${job}",
                                expose=["classification"],
                            ),
                        ],
                    ),
                    TaskNode(
                        task_id="write_report",
                        goal="Write a report from ${analyses}",
                    ),
                ],
                else_body=[
                    FailNode(message="No failures found"),
                ],
            ),
            ReturnNode(expression="analyses"),
        ],
    )


@pytest.fixture
def while_workflow():
    return WorkflowNode(
        body=[
            TaskNode(task_id="check", goal="Check status", expose=["count"]),
            WhileNode(
                expression="count > 0",
                max_iterations=24,
                body=[
                    NotifyNode(message="${count} jobs still running"),
                    WaitNode(duration_seconds=3600),
                    TaskNode(task_id="recheck", goal="Recheck status", expose=["count"]),
                ],
            ),
            NotifyNode(message="All jobs completed"),
        ],
    )


def test_generate_html_empty_workflow():
    workflow = WorkflowNode(body=[])
    result = generate_awl_html(workflow)
    assert "<html" in result
    assert "Empty Workflow" in result


def test_generate_html_simple(simple_workflow):
    result = generate_awl_html(simple_workflow, title="test.awl")
    assert "test.awl" in result
    data = _extract_graph_data(result)
    task_nodes = [n for n in data["nodes"] if n["type"] == "task"]
    assert len(task_nodes) == 1
    assert "check_jobs" in task_nodes[0]["label"]
    assert task_nodes[0]["tooltipData"]["hints"] == ["no-history", "no-kg"]
    assert task_nodes[0]["tooltipData"]["expose"] == ["running_count"]


def test_generate_html_complex(complex_workflow):
    result = generate_awl_html(complex_workflow, title="complex.awl")
    assert "complex.awl" in result
    data = _extract_graph_data(result)

    node_labels = [n["label"] for n in data["nodes"]]
    assert any("find_failures" in lbl for lbl in node_labels)
    assert any("analyze_failure" in lbl for lbl in node_labels)
    assert any("write_report" in lbl for lbl in node_labels)

    decisions = [n for n in data["nodes"] if n["shape"] == "diamond"]
    assert len(decisions) == 1
    assert "failed_jobs" in decisions[0]["label"]

    merges = [n for n in data["nodes"] if n["shape"] == "circle"]
    assert len(merges) == 1

    loop_clusters = [c for c in data["clusters"] if c["type"] == "loop"]
    assert len(loop_clusters) == 1
    assert "failed_jobs" in loop_clusters[0]["label"]
    assert "limit=5" in loop_clusters[0]["label"]
    assert "collect=analyses" in loop_clusters[0]["label"]

    assert len(data["backEdges"]) >= 1

    all_edge_labels = [lbl for e in data["edges"] for lbl in e["labels"]]
    assert "failed_jobs" in all_edge_labels

    assert any(n["type"] == "fail" for n in data["nodes"])
    assert any(n["type"] == "return" for n in data["nodes"])

    find_task = next(n for n in data["nodes"] if "find_failures" in n["label"])
    assert find_task["tooltipData"]["max_tool_calls"] == 50


def test_generate_html_while(while_workflow):
    result = generate_awl_html(while_workflow, title="monitor.awl")
    data = _extract_graph_data(result)

    while_clusters = [c for c in data["clusters"] if c["type"] == "while"]
    assert len(while_clusters) == 1
    assert "count" in while_clusters[0]["label"]
    assert "max_iterations=24" in while_clusters[0]["label"]

    wait_nodes = [n for n in data["nodes"] if n["type"] == "wait"]
    assert len(wait_nodes) == 1
    assert "1h" in wait_nodes[0]["label"]

    notify_nodes = [n for n in data["nodes"] if n["type"] == "notify"]
    assert len(notify_nodes) >= 1

    assert len(data["backEdges"]) >= 1


def test_generate_html_goal_node():
    workflow = WorkflowNode(
        body=[
            GoalNode(
                goal_id="fix_issue",
                success_criteria="All tests pass",
                max_actions=10,
                body=[
                    TaskNode(task_id="diagnose", goal="Find the root cause"),
                    TaskNode(task_id="fix", goal="Apply the fix"),
                ],
            ),
        ]
    )
    result = generate_awl_html(workflow)
    data = _extract_graph_data(result)
    goal_clusters = [c for c in data["clusters"] if c["type"] == "goal"]
    assert len(goal_clusters) == 1
    assert "fix_issue" in goal_clusters[0]["label"]
    assert "max_actions=10" in goal_clusters[0]["label"]


def test_generate_html_set_node():
    workflow = WorkflowNode(body=[SetNode(variable="x", value="hello")])
    result = generate_awl_html(workflow)
    data = _extract_graph_data(result)
    set_nodes = [n for n in data["nodes"] if n["type"] == "set"]
    assert len(set_nodes) == 1
    assert "x" in set_nodes[0]["label"]
    assert "hello" in set_nodes[0]["sublabel"]


def test_generate_html_wait_formats():
    w_seconds = WorkflowNode(body=[WaitNode(duration_seconds=45)])
    data = _extract_graph_data(generate_awl_html(w_seconds))
    assert any("45s" in n["label"] for n in data["nodes"])

    w_minutes = WorkflowNode(body=[WaitNode(duration_seconds=300)])
    data = _extract_graph_data(generate_awl_html(w_minutes))
    assert any("5m" in n["label"] for n in data["nodes"])

    w_hours = WorkflowNode(body=[WaitNode(duration_seconds=7200)])
    data = _extract_graph_data(generate_awl_html(w_hours))
    assert any("2h" in n["label"] for n in data["nodes"])


def test_generate_html_has_legend():
    workflow = WorkflowNode(body=[TaskNode(task_id="t", goal="g")])
    result = generate_awl_html(workflow)
    for key, style in NODE_STYLES.items():
        if key == "merge":
            continue
        assert style["label"] in result


def test_generate_html_header_stats(complex_workflow):
    result = generate_awl_html(complex_workflow, title="stats.awl")
    assert "3" in result
    assert "Max steps" in result
    assert "50" in result


def test_generate_html_input_variables():
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="t", goal="Process ${quarter} of ${year}"),
        ]
    )
    result = generate_awl_html(workflow, title="vars.awl")
    assert "quarter" in result
    assert "year" in result


def test_task_tooltip_data():
    workflow = WorkflowNode(
        body=[
            TaskNode(
                task_id="detailed",
                goal="The goal text",
                context="Some context",
                constraints="Some constraints",
                success="Success criteria",
            ),
        ]
    )
    result = generate_awl_html(workflow)
    data = _extract_graph_data(result)
    task = next(n for n in data["nodes"] if n["type"] == "task")
    td = task["tooltipData"]
    assert td["goal"] == "The goal text"
    assert td["context"] == "Some context"
    assert td["constraints"] == "Some constraints"
    assert td["success"] == "Success criteria"


def test_discover_awl_scripts(tmp_path):
    awl_dir = tmp_path / "workflows"
    awl_dir.mkdir()
    (awl_dir / "test1.awl").write_text("@start\n@end")
    (awl_dir / "test2.awl").write_text("@start\n@end")
    (awl_dir / "not_awl.txt").write_text("nope")

    nested = awl_dir / "sub"
    nested.mkdir()
    (nested / "test3.awl").write_text("@start\n@end")

    with (
        patch("ai_assist.awl_visualization.Path.cwd", return_value=tmp_path),
        patch("ai_assist.awl_visualization.get_config_dir", return_value=tmp_path / "no-config"),
    ):
        scripts = discover_awl_scripts()

    assert len(scripts) == 3
    names = {s.name for s in scripts}
    assert names == {"test1.awl", "test2.awl", "test3.awl"}


def test_discover_awl_scripts_with_skills_cache(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    skills_cache = config_dir / "skills-cache"
    skill_dir = skills_cache / "my-skill" / "scripts"
    skill_dir.mkdir(parents=True)
    (skill_dir / "workflow.awl").write_text("@start\n@end")

    with (
        patch("ai_assist.awl_visualization.Path.cwd", return_value=tmp_path / "empty-cwd"),
        patch("ai_assist.awl_visualization.get_config_dir", return_value=config_dir),
    ):
        scripts = discover_awl_scripts()

    assert len(scripts) == 1
    assert scripts[0].name == "workflow.awl"


def test_discover_awl_scripts_deduplication(tmp_path):
    awl_file = tmp_path / "test.awl"
    awl_file.write_text("@start\n@end")

    with (
        patch("ai_assist.awl_visualization.Path.cwd", return_value=tmp_path),
        patch("ai_assist.awl_visualization.get_config_dir", return_value=tmp_path),
    ):
        scripts = discover_awl_scripts()

    assert len(scripts) == 1


@patch("webbrowser.open")
def test_open_awl_visualization(mock_open, tmp_path):
    awl_file = tmp_path / "test.awl"
    awl_file.write_text("@start\n@task hello\nGoal: Say hello\nExpose: greeting\n@end\n@end")

    with patch("ai_assist.awl_visualization.get_config_dir", return_value=tmp_path):
        filepath = open_awl_visualization(str(awl_file))

    assert filepath.endswith(".html")
    mock_open.assert_called_once()
    content = Path(filepath).read_text()
    assert "<html" in content
    assert "hello" in content
    assert "greeting" in content


@patch("webbrowser.open")
def test_open_awl_visualization_file_not_found(mock_open):
    with pytest.raises(FileNotFoundError):
        open_awl_visualization("/nonexistent/path/script.awl")
    mock_open.assert_not_called()


def test_node_styles_coverage():
    expected = ["task", "if", "loop", "while", "wait", "notify", "set", "return", "fail", "goal"]
    for node_type in expected:
        assert node_type in NODE_STYLES
        assert "color" in NODE_STYLES[node_type]
        assert "bg" in NODE_STYLES[node_type]
        assert "icon" in NODE_STYLES[node_type]
        assert "label" in NODE_STYLES[node_type]


def test_graph_data_structure():
    workflow = WorkflowNode(body=[TaskNode(task_id="t", goal="g")])
    data = _extract_graph_data(generate_awl_html(workflow))
    assert "nodes" in data
    assert "edges" in data
    assert "clusters" in data
    assert "backEdges" in data


def test_variable_flow_on_edges():
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="a", goal="Do A", expose=["x", "y"]),
            TaskNode(task_id="b", goal="Do B"),
        ]
    )
    data = _extract_graph_data(generate_awl_html(workflow))
    edges_with_vars = [e for e in data["edges"] if e["labels"]]
    assert len(edges_with_vars) == 1
    assert set(edges_with_vars[0]["labels"]) == {"x", "y"}


def test_if_creates_decision_and_merge():
    workflow = WorkflowNode(
        body=[
            IfNode(
                expression="x > 0",
                then_body=[TaskNode(task_id="a", goal="A")],
                else_body=[TaskNode(task_id="b", goal="B")],
            )
        ]
    )
    data = _extract_graph_data(generate_awl_html(workflow))
    diamonds = [n for n in data["nodes"] if n["shape"] == "diamond"]
    assert len(diamonds) == 1
    merges = [n for n in data["nodes"] if n["shape"] == "circle"]
    assert len(merges) == 1
    then_edges = [e for e in data["edges"] if "then" in e.get("labels", [])]
    else_edges = [e for e in data["edges"] if "else" in e.get("labels", [])]
    assert len(then_edges) == 1
    assert len(else_edges) == 1


def test_loop_creates_cluster_and_back_edge():
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="items",
                item_var="item",
                body=[TaskNode(task_id="process", goal="Process")],
                collect="results",
            )
        ]
    )
    data = _extract_graph_data(generate_awl_html(workflow))
    assert len(data["clusters"]) == 1
    assert data["clusters"][0]["type"] == "loop"
    assert "items" in data["clusters"][0]["label"]
    assert len(data["backEdges"]) >= 1


def test_cdns_included():
    workflow = WorkflowNode(body=[TaskNode(task_id="t", goal="g")])
    result = generate_awl_html(workflow)
    assert "dagre" in result
    assert "d3" in result


def test_tooltip_div_present():
    workflow = WorkflowNode(body=[TaskNode(task_id="t", goal="g")])
    result = generate_awl_html(workflow)
    assert 'id="tooltip"' in result


def test_nested_if_in_loop():
    workflow = WorkflowNode(
        body=[
            LoopNode(
                collection="jobs",
                item_var="job",
                body=[
                    IfNode(
                        expression="job.status == 'fail'",
                        then_body=[TaskNode(task_id="fix", goal="Fix it")],
                        else_body=[],
                    )
                ],
            )
        ]
    )
    data = _extract_graph_data(generate_awl_html(workflow))
    assert len(data["clusters"]) == 1
    diamonds = [n for n in data["nodes"] if n["shape"] == "diamond"]
    assert len(diamonds) == 1


def test_build_graph_directly():
    workflow = WorkflowNode(
        body=[
            TaskNode(task_id="a", goal="A", expose=["x"]),
            TaskNode(task_id="b", goal="B"),
        ]
    )
    data = _build_graph(workflow)
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    assert data["edges"][0]["labels"] == ["x"]
