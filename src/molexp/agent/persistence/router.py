"""Generic CRUD route factory for :class:`TieredResourceStore` instances.

Lets each resource kind (skills, tools, …) share one canonical set of
HTTP endpoints — list/create/get/update/delete — without re-declaring
the same route handlers per kind. Each kind passes:

- a ``store_factory`` that turns the FastAPI workspace dep into its
  :class:`TieredResourceStore` instance
- a ``spec_to_response`` adapter that converts the kind's
  :class:`ResourceSpec` subclass into the API response model
- ``create_kwargs`` / ``update_kwargs`` adapters that translate the
  kind's request models into ``Store.create``/``update`` kwargs
- the kind-specific request/response Pydantic classes (FastAPI uses
  these for both request validation and OpenAPI schema generation)
- ``prefix`` (e.g. ``"/skills"``), ``tags`` (for Swagger grouping),
  and ``list_field`` (the attribute name inside the list-response
  envelope, e.g. ``"skills"`` or ``"tools"``)

The factory returns a fully-formed :class:`fastapi.APIRouter` ready to
:meth:`include_router` onto the parent agent admin router.

Validation flow stays uniform: ``ValueError``s from the store map to
HTTP 400, ``KeyError``s map to HTTP 404. Approval-policy gating and
cross-kind concerns (e.g. MCP secret resolution) stay in the kind's
own dedicated routes — this factory is for the resource-shape CRUD
that's *common* across kinds.
"""

from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from molexp.agent.persistence.tiered import ResourceSpec, Scope, TieredResourceStore

SpecT = TypeVar("SpecT", bound=ResourceSpec)
ItemResponseT = TypeVar("ItemResponseT", bound=BaseModel)
ListResponseT = TypeVar("ListResponseT", bound=BaseModel)
CreateRequestT = TypeVar("CreateRequestT", bound=BaseModel)
UpdateRequestT = TypeVar("UpdateRequestT", bound=BaseModel)


class _SimpleMessage(BaseModel):
    """Minimal acknowledgement payload for delete responses."""

    message: str


def tiered_router_factory[
    SpecT: ResourceSpec,
    ItemResponseT: BaseModel,
    ListResponseT: BaseModel,
    CreateRequestT: BaseModel,
    UpdateRequestT: BaseModel,
](
    *,
    store_factory: Callable[[Any], TieredResourceStore[SpecT]],
    spec_to_response: Callable[[SpecT], ItemResponseT],
    item_response_cls: type[ItemResponseT],
    list_response_cls: type[ListResponseT],
    create_request_cls: type[CreateRequestT],
    update_request_cls: type[UpdateRequestT],
    create_kwargs: Callable[[CreateRequestT], dict[str, Any]],
    update_kwargs: Callable[[UpdateRequestT], dict[str, Any]],
    workspace_dependency: Callable[..., Any],
    prefix: str,
    tags: list[str],
    list_field: str,
) -> APIRouter:
    """Build a FastAPI router exposing CRUD over a tiered resource store.

    Args:
        store_factory: Construct the kind's
            :class:`TieredResourceStore` from a FastAPI workspace
            dependency.
        spec_to_response: Convert one stored spec into the kind's
            response model.
        item_response_cls: Pydantic class returned for single-resource
            endpoints.
        list_response_cls: Pydantic class wrapping the ``list_all``
            payload; must accept ``{<list_field>: list[item_response]}``.
        create_request_cls: Request model for the ``POST /`` endpoint.
        update_request_cls: Request model for the ``PATCH /{id}`` endpoint.
        create_kwargs: Translate a validated create request into the
            kwargs the store's ``create`` accepts. Must include
            ``scope`` (``"user"`` or ``"workspace"``).
        update_kwargs: Translate a validated update request into the
            kwargs the store's ``update`` accepts. Should drop ``None``
            values so callers can omit fields they don't want to change.
        workspace_dependency: FastAPI dependency yielding the workspace
            (typically ``get_workspace``).
        prefix: Route prefix mounted under the parent (e.g. ``"/skills"``).
        tags: OpenAPI/Swagger tag list for the routes.
        list_field: Name of the field inside ``list_response_cls`` that
            holds the array of items (e.g. ``"skills"``).

    Returns:
        Fully-formed :class:`APIRouter` ready to mount via
        :meth:`include_router` on the agent admin router.
    """
    # FastAPI's ``tags`` accepts ``list[str | Enum] | None``; widen the
    # caller's plain ``list[str]`` to match the SDK's invariant generic.
    router_tags: list[str | Enum] = list(tags)
    router = APIRouter(prefix=prefix, tags=router_tags)

    @router.get("", response_model=list_response_cls)
    async def list_resources(
        workspace=Depends(workspace_dependency),
    ) -> ListResponseT:
        store = store_factory(workspace)
        items = [spec_to_response(s) for s in store.list_all()]
        return list_response_cls(**{list_field: items})

    @router.post("", response_model=item_response_cls, status_code=201)
    async def create_resource(
        request: CreateRequestT,
        workspace=Depends(workspace_dependency),
    ) -> ItemResponseT:
        store = store_factory(workspace)
        kwargs = dict(create_kwargs(request))
        scope_value = kwargs.pop("scope", Scope.WORKSPACE.value)
        try:
            scope = Scope(scope_value) if isinstance(scope_value, str) else scope_value
            spec = store.create(scope, **kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return spec_to_response(spec)

    @router.get("/{resource_id}", response_model=item_response_cls)
    async def get_resource(
        resource_id: str,
        workspace=Depends(workspace_dependency),
    ) -> ItemResponseT:
        store = store_factory(workspace)
        spec = store.get(resource_id)
        if spec is None:
            raise HTTPException(
                status_code=404,
                detail=f"id '{resource_id}' not found",
            )
        return spec_to_response(spec)

    @router.patch("/{resource_id}", response_model=item_response_cls)
    async def update_resource(
        resource_id: str,
        request: UpdateRequestT,
        scope: Literal["user", "workspace"] = Query(
            "workspace",
            description="Tier the entry belongs to.",
        ),
        workspace=Depends(workspace_dependency),
    ) -> ItemResponseT:
        store = store_factory(workspace)
        try:
            spec = store.update(
                resource_id,
                scope=Scope(scope),
                **update_kwargs(request),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return spec_to_response(spec)

    @router.delete("/{resource_id}", response_model=_SimpleMessage)
    async def delete_resource(
        resource_id: str,
        scope: Literal["user", "workspace"] = Query(
            "workspace",
            description="Tier to delete from.",
        ),
        workspace=Depends(workspace_dependency),
    ) -> _SimpleMessage:
        store = store_factory(workspace)
        deleted = store.delete(resource_id, scope=Scope(scope))
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"id '{resource_id}' not found at scope '{scope}'.",
            )
        return _SimpleMessage(message=f"deleted from {scope}")

    return router


__all__ = ["tiered_router_factory"]
