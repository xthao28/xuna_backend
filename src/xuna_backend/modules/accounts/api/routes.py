from fastapi import APIRouter

from xuna_backend.modules.accounts.api.dependencies import CurrentOwner

router = APIRouter(prefix="/v1", tags=["accounts"])


@router.get("/me")
async def me(owner: CurrentOwner) -> dict[str, str]:
    """The authenticated owner's XUNA `userId` (never the provider subject)."""
    return {"userId": str(owner.owner_id)}
