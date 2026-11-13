"""OpenAPI 문서에 /ask 의 SSE 이벤트를 싣는다 — FastAPI 는 StreamingResponse 가 흘리는 이벤트 모양을 모른다.

이벤트 한 건은 OpenAPI 3.2 의 SSE 꼴로 적는다: itemSchema 가 `event`(이름) · `data`(JSON 문자열)이고, data 의 JSON
스키마는 contentSchema 로 components 의 모델을 가리킨다. FastAPI 가 자기 SSE 응답(EventSourceResponse)에 쓰는 꼴과 같다.
"""

import copy
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

SSE_MEDIA_TYPE = "text/event-stream"
_REF_TEMPLATE = "#/components/schemas/{model}"  # FastAPI 가 components 를 가리키는 꼴


def sse_response(events: Mapping[str, type[BaseModel]], *, description: str) -> dict[str, Any]:
    """라우터 responses 칸 하나 — text/event-stream 의 이벤트 한 건이 이벤트 이름 하나와 그 data 모델이다."""
    refs, _ = _schemas(events)
    items = [_event_item(name, refs[model]) for name, model in events.items()]
    return {
        "description": description,
        "content": {SSE_MEDIA_TYPE: {"itemSchema": {"oneOf": items}}},
    }


def with_event_schemas(
    openapi: Mapping[str, Any], events: Mapping[str, type[BaseModel]]
) -> dict[str, Any]:
    """components.schemas 에 이벤트 data 모델과 그 안의 모델을 더한 OpenAPI 사본. FastAPI 처럼 이름순으로 둔다.

    Raises:
      ValueError: 이미 있는 컴포넌트와 이름이 같은데 스키마가 다를 때.
    """
    _, defs = _schemas(events)
    result = copy.deepcopy(dict(openapi))
    schemas: dict[str, Any] = result.setdefault("components", {}).setdefault("schemas", {})
    for name, schema in defs.items():
        if schemas.get(name, schema) != schema:
            raise ValueError(f"event schema {name!r} collides with an existing component")
        schemas[name] = schema
    result["components"]["schemas"] = dict(sorted(schemas.items()))
    return result


def _schemas(
    events: Mapping[str, type[BaseModel]],
) -> tuple[dict[type[BaseModel], dict[str, Any]], dict[str, Any]]:
    """이벤트 data 모델마다 components 를 가리키는 $ref 와, components 에 넣을 스키마 (이름 → 스키마)."""
    keyed, top = models_json_schema(
        [(model, "serialization") for model in events.values()], ref_template=_REF_TEMPLATE
    )
    refs = {model: dict(schema) for (model, _), schema in keyed.items()}
    return refs, dict(top.get("$defs", {}))


def _event_item(name: str, ref: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["event", "data"],
        "properties": {
            "event": {"const": name},
            "data": {
                "type": "string",
                "contentMediaType": "application/json",
                "contentSchema": dict(ref),
            },
        },
    }
