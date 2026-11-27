"""Allow GitHub collaboration while reserving repository writes for inspected Git."""

import json
import re
from urllib.parse import parse_qsl, unquote

from graphql import FieldNode, FragmentDefinitionNode, GraphQLError, OperationDefinitionNode, OperationType, parse

_REPO = r"/repos/[\w.-]+/[\w.-]+"
_REST_WRITES = {
    "POST": (
        r"/markdown",
        _REPO + r"/pulls",
        _REPO + r"/issues/\d+/comments",
        _REPO + r"/pulls/\d+/(?:comments|reviews)",
        _REPO + r"/pulls/\d+/comments/\d+/replies",
        _REPO + r"/pulls/\d+/reviews/\d+/(?:events|comments)",
    ),
    "PATCH": (
        _REPO + r"/pulls/\d+",
        _REPO + r"/(?:issues|pulls)/comments/\d+",
    ),
    "PUT": (_REPO + r"/pulls/\d+/reviews/\d+(?:/dismissals)?",),
    "DELETE": (_REPO + r"/(?:issues|pulls)/comments/\d+", _REPO + r"/pulls/\d+/reviews/\d+"),
}
_EDITORIAL_MUTATIONS = {
    "addComment",
    "addPullRequestReview",
    "addPullRequestReviewComment",
    "addPullRequestReviewThread",
    "addPullRequestReviewThreadReply",
    "closePullRequest",
    "convertPullRequestToDraft",
    "createPullRequest",
    "deleteIssueComment",
    "deletePullRequestReview",
    "deletePullRequestReviewComment",
    "dismissPullRequestReview",
    "markPullRequestReadyForReview",
    "reopenPullRequest",
    "resolveReviewThread",
    "submitPullRequestReview",
    "unresolveReviewThread",
    "updateIssueComment",
    "updatePullRequest",
    "updatePullRequestReview",
    "updatePullRequestReviewComment",
}
_METHOD_HEADERS = {
    b"x-http-method-override",
    b"x-http-method",
    b"x-method-override",
    b"x-original-url",
    b"x-rewrite-url",
}


def _unique_keys(pairs: list) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def _allowed_graphql(body: bytes) -> bool:
    try:
        payload = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_keys)
        if not isinstance(payload, dict) or payload.keys() - {"query", "variables", "operationName"}:
            return False
        query = payload.get("query")
        if not isinstance(query, str):
            return False
        document = parse(query, no_location=True, max_tokens=20000)
    except ValueError, GraphQLError, RecursionError:
        return False
    for definition in document.definitions:
        if isinstance(definition, FragmentDefinitionNode):
            continue
        if not isinstance(definition, OperationDefinitionNode):
            return False
        if definition.operation == OperationType.QUERY:
            continue
        if definition.operation != OperationType.MUTATION or not all(
            isinstance(field, FieldNode) and field.name.value in _EDITORIAL_MUTATIONS
            for field in definition.selection_set.selections
        ):
            return False
    return True


def github_request_allowed(host: str, method: str, target: str, headers: tuple, media_type: str, body: bytes) -> bool:
    """Check the operation before the inspected request can open an upstream connection."""
    path, _, query = target.partition("?")
    if any(name.lower() in _METHOD_HEADERS for name, _ in headers):
        return False
    if any(key.lower() == "_method" for key, _ in parse_qsl(query)):
        return False
    if method in ("GET", "HEAD"):
        return not body and unquote(path).rstrip("/").lower() != "/graphql"
    if query or any(part in (".", "..") for part in path.split("/")):
        return False
    if host == "github.com":
        git = re.fullmatch(r"/[\w.-]+/[\w.-]+/(git-upload-pack|git-receive-pack)", path, re.ASCII)
        return bool(git and method == "POST" and media_type == f"application/x-{git[1]}-request")
    if host != "api.github.com" or media_type != "application/json" and (body or method != "DELETE"):
        return False
    if path == "/graphql":
        return method == "POST" and _allowed_graphql(body)
    return any(re.fullmatch(route, path, re.ASCII) for route in _REST_WRITES.get(method, ()))
