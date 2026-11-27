import json

import pytest

from ai_assist.github_egress import github_request_allowed


@pytest.mark.parametrize(
    "method,path,allowed",
    [
        ("GET", "/repos/team/project/git/refs/heads/main", True),
        ("GET", "/repos/team/project/pulls/2/comments?page=2", True),
        ("POST", "/repos/team/project/pulls", True),
        ("PATCH", "/repos/team/project/pulls/2", True),
        ("POST", "/repos/team/project/issues/2/comments", True),
        ("POST", "/repos/team/project/pulls/2/reviews", True),
        ("PATCH", "/repos/team/project/pulls/comments/7", True),
        ("POST", "/repos/team/project/git/blobs", False),
        ("POST", "/repos/team/project/git/trees", False),
        ("POST", "/repos/team/project/git/commits", False),
        ("PATCH", "/repos/team/project/git/refs/heads/main", False),
        ("PUT", "/repos/team/project/contents/example", False),
        ("PUT", "/repos/team/project/pulls/2/merge", False),
        ("POST", "/repos/team/project/dispatches", False),
    ],
)
def test_rest_collaboration(method, path, allowed):
    body = b"" if method == "GET" else b"{}"
    assert github_request_allowed("api.github.com", method, path, (), "application/json", body) is allowed


@pytest.mark.parametrize(
    "payload,allowed",
    [
        ({"query": "query { viewer { login } }"}, True),
        ({"query": "mutation($i: AddCommentInput!) { posted: addComment(input: $i) { clientMutationId } }"}, True),
        (
            {
                "query": "mutation { a: addComment(input: {}) { clientMutationId } b: updateRef(input: {}) { ref { id } } }"
            },
            False,
        ),
        (
            {
                "query": "mutation { ...Writes } fragment Writes on Mutation { updateRef(input: {}) { clientMutationId } }"
            },
            False,
        ),
        ({"extensions": {}}, False),
        ([{"query": "query { viewer { login } }"}], False),
    ],
)
def test_graphql_operation_selection(payload, allowed):
    assert (
        github_request_allowed(
            "api.github.com", "POST", "/graphql", (), "application/json", json.dumps(payload).encode()
        )
        is allowed
    )


def test_git_transport_and_web_writes():
    for service in ("git-upload-pack", "git-receive-pack"):
        assert github_request_allowed(
            "github.com", "POST", f"/o/r.git/{service}", (), f"application/x-{service}-request", b"0000"
        )
    assert not github_request_allowed("github.com", "POST", "/o/r/edit/main/file", (), "application/json", b"{}")
    assert not github_request_allowed(
        "api.github.com", "GET", "/repos/o/r", ((b"X-HTTP-Method-Override", b"PUT"),), "", b""
    )
