import pandas as pd

def get_business_answer(question, sales_df, inventory_df):

    question = question.lower().strip()

    if sales_df is None or inventory_df is None:
        return "Please upload both Sales and Inventory datasets first."

    sales = sales_df.copy()
    inventory = inventory_df.copy()

    if "Revenue" in sales.columns:
        sales["Revenue"] = pd.to_numeric(
            sales["Revenue"], errors="coerce"
        ).fillna(0)

    if "Profit" in sales.columns:
        sales["Profit"] = pd.to_numeric(
            sales["Profit"], errors="coerce"
        ).fillna(0)

    if "Quantity" in sales.columns:
        sales["Quantity"] = pd.to_numeric(
            sales["Quantity"], errors="coerce"
        ).fillna(0)

    if "Available_Stock" in inventory.columns:
        inventory["Available_Stock"] = pd.to_numeric(
            inventory["Available_Stock"], errors="coerce"
        ).fillna(0)

    if "Reorder_Level" in inventory.columns:
        inventory["Reorder_Level"] = pd.to_numeric(
            inventory["Reorder_Level"], errors="coerce"
        ).fillna(0)

    # TOTAL SALES
    if any(word in question for word in ["total sales", "revenue"]):
        return f"Your total revenue is ₹{sales['Revenue'].sum():,.2f}."

    # TOTAL PROFIT
    if "profit" in question and "highest" not in question:
        return f"Your total profit is ₹{sales['Profit'].sum():,.2f}."

    # TOTAL ORDERS
    if "orders" in question:
        if "Order_ID" in sales.columns:
            return f"Total orders: {sales['Order_ID'].nunique()}."
        return "Order_ID column is not available."

    # TOP SELLING PRODUCT
    if any(word in question for word in [
    "top selling",
    "best selling",
    "highest sales",
    "sells the most",
    "sell the most",
    "most sold",
    "which product sells",
    "which product is selling"
]):
        if {"Product_Name", "Quantity"}.issubset(sales.columns):
            result = sales.groupby("Product_Name")["Quantity"].sum()
            product = result.idxmax()
            return f"Your top-selling product is {product}, with {result.max():,.0f} units sold."

    # HIGHEST PROFIT PRODUCT
    if "highest profit" in question or "most profitable" in question:
        result = sales.groupby("Product_Name")["Profit"].sum()
        return f"Most profitable product: {result.idxmax()} (₹{result.max():,.2f})."

    # LOW STOCK PRODUCTS
    if any(word in question for word in ["low stock", "restock", "reorder"]):
        low = inventory[
            inventory["Available_Stock"] <= inventory["Reorder_Level"]
        ]

        if low.empty:
            return "All products currently have sufficient stock."

        names = ", ".join(low["Product_Name"].astype(str).tolist())
        return f"Products needing replenishment: {names}."

    # OUT OF STOCK
    if "out of stock" in question:
        out = inventory[inventory["Available_Stock"] <= 0]

        if out.empty:
            return "No products are currently out of stock."

        return "Out of stock products: " + ", ".join(
            out["Product_Name"].astype(str).tolist()
        )

    # TOTAL PRODUCTS
    if "total products" in question:
        return f"Total products: {inventory['Product_ID'].nunique()}."

    # LOWEST PROFIT
    if "lowest profit" in question:
        result = sales.groupby("Product_Name")["Profit"].sum()
        return f"Lowest-profit product: {result.idxmin()} (₹{result.min():,.2f})."

    return (
        "I can answer questions about total sales, profit, orders, "
        "top-selling products, low stock and out-of-stock products. "
        "Please try one of these questions."
    )