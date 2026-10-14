from pydantic import BaseModel, Field

from tau2.domains.retail_stock.data_model import StockEvent, StockRetailDB
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class StockCheck(BaseModel):
    """Result of a fixed inventory lookup."""

    product_id: str
    item_id: str
    stock_quantity: int = Field(ge=0)
    stock_version: int = Field(ge=1)
    simulation_time: int = Field(ge=0)
    events_since_previous_query: list[StockEvent] = Field(default_factory=list)


class RetailStockTools(ToolKitBase):
    """Minimal read-only toolkit for the fixed-inventory experiment."""

    db: StockRetailDB

    def __init__(self, db: StockRetailDB) -> None:
        super().__init__(db)
        self._query_count: dict[tuple[str, str], int] = {}
        self._last_seen_version: dict[tuple[str, str], int] = {}

    def _get_stock_quantity(self, product_id: str, item_id: str) -> int:
        """Resolve and validate a product variant's stock quantity."""

        product = self.db.products.get(product_id)
        if product is None:
            raise ValueError("Product not found")
        variant = product.variants.get(item_id)
        if variant is None:
            raise ValueError("Variant not found")
        return variant.stock_quantity

    def _apply_background_purchase(self, product_id: str, item_id: str) -> None:
        """Apply the phase-two event once, immediately after the first lookup."""

        if self.db.event_log:
            return
        variant = self.db.products[product_id].variants[item_id]
        before = variant.stock_quantity
        variant.stock_quantity -= 6
        old_version = variant.stock_version
        variant.stock_version += 1
        variant.simulation_time += 1
        self.db.event_log.append(
            StockEvent(
                event_id="background-purchase-after-first-query",
                event_type="background_purchase",
                product_id=product_id,
                item_id=item_id,
                quantity_delta=-6,
                stock_before=before,
                stock_after=variant.stock_quantity,
                version_before=old_version,
                version_after=variant.stock_version,
                simulation_time=variant.simulation_time,
            )
        )

    @is_tool(ToolType.READ, mutates_state=True)
    def get_variant_stock(self, product_id: str, item_id: str) -> StockCheck:
        """Get the current stock count and version for a product variant.

        Stock can change after any lookup. A lookup is evidence at its reported
        simulation time only; it does not reserve stock or create an order.

        Args:
            product_id: Product identifier.
            item_id: Variant/item identifier belonging to the product.

        Returns:
            The identifiers and current fixed stock quantity.
        """

        key = (product_id, item_id)
        variant = self.db.products[product_id].variants[item_id]
        last_seen = self._last_seen_version.get(key, 0)
        result = StockCheck(
            product_id=product_id,
            item_id=item_id,
            stock_quantity=self._get_stock_quantity(product_id, item_id),
            stock_version=variant.stock_version,
            simulation_time=variant.simulation_time,
            events_since_previous_query=[
                event
                for event in self.db.event_log
                if event.product_id == product_id
                and event.item_id == item_id
                and event.version_after > last_seen
            ],
        )
        self._last_seen_version[key] = variant.stock_version
        self._query_count[key] = self._query_count.get(key, 0) + 1
        if self._query_count[key] == 1:
            self._apply_background_purchase(product_id, item_id)
        return result
