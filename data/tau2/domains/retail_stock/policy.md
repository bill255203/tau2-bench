# Fixed inventory policy

- Use `get_variant_stock` before answering any question about a requested quantity.
- Inventory can change between turns. If the customer asks you to confirm supply,
  query again and use the newest version and simulated time as your evidence.
- Compare the returned `stock_quantity` with the customer's requested quantity.
- State the available quantity and, when insufficient, the exact shortfall.
- Inventory checks do not reserve stock. Never claim that inventory was reserved or that an order was placed.
- This experiment has no reservation or ordering capability.
