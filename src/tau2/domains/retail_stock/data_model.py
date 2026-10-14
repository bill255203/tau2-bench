from typing import Any, Literal

from pydantic import BaseModel, Field

from tau2.domains.retail.data_model import Product, RetailDB, Variant
from tau2.domains.retail_stock.utils import RETAIL_STOCK_DB_PATH


class StockVariant(Variant):
    """A retail variant with a fixed, directly observable stock count."""

    stock_quantity: int = Field(ge=0, description="Units currently in stock")
    stock_version: int = Field(default=1, ge=1)
    simulation_time: int = Field(default=0, ge=0)


class StockEvent(BaseModel):
    """An auditable, deterministic inventory change in simulated time."""

    event_id: str
    event_type: Literal["background_purchase"]
    product_id: str
    item_id: str
    quantity_delta: int
    stock_before: int = Field(ge=0)
    stock_after: int = Field(ge=0)
    version_before: int = Field(ge=1)
    version_after: int = Field(ge=1)
    simulation_time: int = Field(ge=0)


class StockProduct(Product):
    """A product whose variants include fixed stock quantities."""

    variants: dict[str, StockVariant]


class StockRetailDB(RetailDB):
    """Retail database used only by the fixed-stock experiment."""

    products: dict[str, StockProduct]
    event_log: list[StockEvent] = Field(default_factory=list)

    def get_statistics(self) -> dict[str, Any]:
        statistics = super().get_statistics()
        statistics["total_stock_quantity"] = sum(
            variant.stock_quantity
            for product in self.products.values()
            for variant in product.variants.values()
        )
        return statistics


def get_db() -> StockRetailDB:
    """Load the isolated fixed-stock database."""

    return StockRetailDB.load(RETAIL_STOCK_DB_PATH)
