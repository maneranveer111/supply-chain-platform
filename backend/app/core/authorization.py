from fastapi import HTTPException, status

from app.core.dependencies import CurrentUser, Role
from app.models.store import Store


def verify_store_access(store: Store, current_user: CurrentUser) -> None:
    """
    Verifies that the current_user is authorized to access the given store.
    
    Rules (Phase 3):
    1. Admin can access ANY store.
    2. Normal user can access a store ONLY if store.user_id == current_user.id.
    3. Demo/system stores (store.user_id IS NULL) are accessible ONLY by Admins.
    
    Raises:
        HTTPException(403) if access is denied.
    """
    # Admins have global access
    if current_user.role == Role.ADMIN.value:
        return

    # User owns the store
    if store.user_id is not None and store.user_id == int(current_user.user_id):
        return

    # All other cases: Access denied
    # This includes another user's store, or a NULL-owned system store.
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You do not have permission to access this store.",
    )
