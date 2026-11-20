
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =====================================================
# 1. APP CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SmartStock AI",
    page_icon="📦",
    layout="wide"
)

# =====================================================
# 2. SESSION STATE
# =====================================================

defaults = {
    "sales_df": None,
    "inventory_df": None,
    "theme": "Dark"
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =====================================================
# 3. PREMIUM THEME
# =====================================================

st.sidebar.markdown("## ⚙️ Appearance")

theme = st.sidebar.radio(
    "Choose Theme",
    ["Dark", "Light"],
    index=0 if st.session_state.theme == "Dark" else 1,
    horizontal=True
)

st.session_state.theme = theme

if theme == "Dark":
    bg = "#0B1426"
    sidebar_bg = "#101F38"
    card_bg = "#172943"
    text_color = "#F1F5F9"
    muted_color = "#A8B8CC"
    border_color = "#263D5B"
    chart_template = "plotly_dark"
else:
    bg = "#F2F6FC"
    sidebar_bg = "#FFFFFF"
    card_bg = "#FFFFFF"
    text_color = "#172B4D"
    muted_color = "#64748B"
    border_color = "#DCE6F2"
    chart_template = "plotly_white"

st.markdown(f"""
<style>
.stApp {{
    background: {bg};
    color: {text_color};
}}

[data-testid="stSidebar"] {{
    background: {sidebar_bg};
    border-right: 1px solid {border_color};
}}

.block-container {{
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}}

h1, h2, h3, h4, p, label {{
    color: {text_color};
}}

[data-testid="stMetric"] {{
    background: {card_bg};
    border: 1px solid {border_color};
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.10);
    transition: 0.25s ease;
}}

[data-testid="stMetric"]:hover {{
    border-color: #06B6D4;
    transform: translateY(-3px);
}}

[data-testid="stMetricLabel"] {{
    color: {muted_color};
}}

[data-testid="stMetricValue"] {{
    color: #38A5FF;
    font-weight: 700;
}}

.stButton > button,
.stDownloadButton > button {{
    background: linear-gradient(90deg, #1769FF, #06B6D4);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    transition: 0.25s ease;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(23,105,255,0.30);
    color: white;
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid {border_color};
    border-radius: 12px;
}}

div[data-testid="stPlotlyChart"] {{
    background: {card_bg};
    border: 1px solid {border_color};
    border-radius: 14px;
    padding: 8px;
}}

.stTextInput input,
.stSelectbox div[data-baseweb="select"],
.stDateInput input {{
    border-radius: 9px;
}}

hr {{
    border-color: {border_color};
}}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 4. COLUMN MAPPING
# =====================================================

COLUMN_MAPPING = {
    "Order_ID": ["order_id", "orderid", "order number"],
    "Order_Date": ["order_date", "date", "sales_date"],
    "Product_ID": ["product_id", "productid", "sku"],
    "Product_Name": ["product_name", "product", "item_name"],
    "Category": ["category", "product_category"],
    "Quantity": ["quantity", "qty", "units_sold"],
    "Revenue": ["revenue", "sales", "sales_amount", "total_sales", "amount"],
    "Profit": ["profit", "profit_amount"],
    "Unit_Price": ["unit_price", "price", "selling_price"],
    "Unit_Cost": ["unit_cost", "cost_price"],
    "Region": ["region", "location", "area"],
    "Available_Stock": ["available_stock", "stock", "current_stock", "inventory"],
    "Reorder_Level": ["reorder_level", "minimum_stock", "min_stock"]
}

def standardize_columns(df):
    df = df.copy()
    rename_dict = {}

    for standard_name, aliases in COLUMN_MAPPING.items():
        for col in df.columns:
            normalized = str(col).strip().lower().replace(" ", "_")

            if normalized == standard_name.lower() or normalized in aliases:
                rename_dict[col] = standard_name
                break

    return df.rename(columns=rename_dict)

def convert_numeric(df, columns):
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""),
                errors="coerce"
            ).fillna(0)

    return df

def missing_columns(df, required):
    return [col for col in required if col not in df.columns]

def money(value):
    return f"₹{value:,.2f}"

def read_file(file):
    if file is None:
        return None

    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)

    except Exception as e:
        st.error(f"Unable to read file: {e}")
        return None

# =====================================================
# 5. EXCEL REPORT
# =====================================================

def make_excel_report(sales, inventory):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if sales is not None:
            sales.to_excel(
                writer,
                sheet_name="Sales Data",
                index=False
            )

        if inventory is not None:
            inventory.to_excel(
                writer,
                sheet_name="Inventory Data",
                index=False
            )

    output.seek(0)
    return output

# =====================================================
# 6. BUSINESS ASSISTANT
# =====================================================

def business_answer(question, sales, inventory):

    q = question.lower().strip()

    if sales is None:
        return "Please upload your Sales Dataset first."

    sales = standardize_columns(sales)
    sales = convert_numeric(sales, ["Revenue", "Profit", "Quantity"])

    # TOP SELLING PRODUCT
    if any(word in q for word in [
        "top product",
        "best selling",
        "top selling",
        "sells the most",
        "sell the most",
        "most sold",
        "which product"
    ]):
        if "Product_Name" not in sales.columns:
            return "Product Name column is missing."

        if "Quantity" not in sales.columns:
            return "Quantity column is missing."

        top = sales.groupby("Product_Name")["Quantity"].sum()

        if top.empty:
            return "No product sales data found."

        return (
            f"🏆 Top-selling product is {top.idxmax()} "
            f"with {top.max():,.0f} units sold."
        )

    # TOTAL REVENUE
    if "revenue" in q or "total sales" in q:
        return f"Total Revenue is {money(sales['Revenue'].sum())}."

    # TOTAL PROFIT
    if "profit" in q:
        return f"Total Profit is {money(sales['Profit'].sum())}."

    # TOTAL ORDERS
    if "orders" in q:
        if "Order_ID" in sales.columns:
            return f"Total Orders: {sales['Order_ID'].nunique()}."
        return "Order ID column is missing."

    # LOW STOCK
    if "low stock" in q or "reorder" in q:
        if inventory is None:
            return "Please upload Inventory Dataset."

        inventory = standardize_columns(inventory)
        inventory = convert_numeric(
            inventory, ["Available_Stock", "Reorder_Level"]
        )

        low = inventory[
            inventory["Available_Stock"] <= inventory["Reorder_Level"]
        ]

        if low.empty:
            return "All products have sufficient stock."

        return "Products needing restock: " + ", ".join(
            low["Product_Name"].astype(str).tolist()
        )

    return (
        "Try asking about Total Revenue, Total Profit, "
        "Top Selling Product, Total Orders or Low Stock."
    )
# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("# 📦 SmartStock AI")
st.sidebar.caption("Business Intelligence Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Home",
        "Upload Data",
        "Business Dashboard",
        "Smart Inventory",
        "AI Demand Forecasting",
        "AI Business Assistant",
        "Reports & Downloads"
    ]
)

# =====================================================
# 8. HOME
# =====================================================

if page == "Home":

    st.title("📦 SmartStock AI")
    st.subheader("AI-Powered Inventory & Demand Intelligence System")

    st.markdown("---")

    st.write(
        "Analyze sales, monitor inventory and make "
        "smarter business decisions using your business data."
    )

    c1, c2, c3 = st.columns(3)

    c1.info(
        "📊 **Sales Analytics**\n\n"
        "Understand business performance and revenue trends."
    )

    c2.info(
        "📦 **Inventory Intelligence**\n\n"
        "Monitor stock levels and identify products to reorder."
    )

    c3.info(
        "🤖 **AI Business Assistant**\n\n"
        "Get quick answers from your uploaded business data."
    )

    st.markdown("---")

    st.subheader("📌 Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Sales Dataset",
            "Uploaded" if st.session_state.sales_df is not None else "Not Uploaded"
        )

    with col2:
        st.metric(
            "Inventory Dataset",
            "Uploaded" if st.session_state.inventory_df is not None else "Not Uploaded"
        )

    st.markdown("---")
    st.subheader("🚀 Get Started")
    st.write("Upload your Sales and Inventory datasets to explore insights.")

    st.caption("SmartStock AI | Data Analytics Project")

# =====================================================
# 9. UPLOAD DATA
# =====================================================

elif page == "Upload Data":

    st.title("📂 Upload Business Data")
    st.write("Upload your Sales and Inventory files.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Sales Dataset")

        sales_file = st.file_uploader(
            "Choose Sales File",
            type=["csv", "xlsx"],
            key="sales_upload"
        )

        if sales_file is not None:
            sales_data = read_file(sales_file)

            if sales_data is not None:
                st.session_state.sales_df = standardize_columns(sales_data)
                st.success("Sales Dataset uploaded successfully!")

    with col2:
        st.subheader("📦 Inventory Dataset")

        inventory_file = st.file_uploader(
            "Choose Inventory File",
            type=["csv", "xlsx"],
            key="inventory_upload"
        )

        if inventory_file is not None:
            inventory_data = read_file(inventory_file)

            if inventory_data is not None:
                st.session_state.inventory_df = standardize_columns(
                    inventory_data
                )
                st.success("Inventory Dataset uploaded successfully!")

    st.markdown("---")

    if st.session_state.sales_df is not None:
        st.subheader("Sales Data Preview")
        st.dataframe(
            st.session_state.sales_df.head(10),
            use_container_width=True
        )

    if st.session_state.inventory_df is not None:
        st.subheader("Inventory Data Preview")
        st.dataframe(
            st.session_state.inventory_df.head(10),
            use_container_width=True
        )

# =====================================================
# 10. BUSINESS DASHBOARD
# =====================================================

elif page == "Business Dashboard":

    st.title("📊 Business Analytics Dashboard")

    sales = st.session_state.sales_df

    if sales is None:
        st.warning("Please upload Sales Dataset first.")

    else:
        sales = standardize_columns(sales)
        sales = convert_numeric(
            sales,
            ["Revenue", "Profit", "Quantity"]
        )

        required = ["Revenue", "Profit", "Quantity"]
        missing = missing_columns(sales, required)

        if missing:
            st.error("Missing Sales Columns: " + ", ".join(missing))

        else:

            filtered = sales.copy()

            # ---------------- FILTERS ----------------

            st.subheader("🔎 Dashboard Filters")

            filter_cols = st.columns(3)

            if "Order_Date" in filtered.columns:
                filtered["Order_Date"] = pd.to_datetime(
                    filtered["Order_Date"],
                    errors="coerce"
                )

                valid_dates = filtered["Order_Date"].dropna()

                if not valid_dates.empty:
                    with filter_cols[0]:
                        date_range = st.date_input(
                            "Order Date Range",
                            value=(
                                valid_dates.min().date(),
                                valid_dates.max().date()
                            )
                        )

                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        filtered = filtered[
                            (filtered["Order_Date"].dt.date >= date_range[0])
                            & (filtered["Order_Date"].dt.date <= date_range[1])
                        ]

            if "Category" in filtered.columns:
                with filter_cols[1]:
                    categories = ["All"] + sorted(
                        filtered["Category"].dropna().astype(str).unique().tolist()
                    )

                    selected_category = st.selectbox(
                        "Category",
                        categories
                    )

                if selected_category != "All":
                    filtered = filtered[
                        filtered["Category"].astype(str) == selected_category
                    ]

            if "Product_Name" in filtered.columns:
                with filter_cols[2]:
                    products = ["All"] + sorted(
                        filtered["Product_Name"].dropna().astype(str).unique().tolist()
                    )

                    selected_product = st.selectbox(
                        "Product",
                        products
                    )

                if selected_product != "All":
                    filtered = filtered[
                        filtered["Product_Name"].astype(str) == selected_product
                    ]

            if filtered.empty:
                st.warning("No records found for selected filters.")

            else:
                # ---------------- KPI CARDS ----------------

                total_revenue = filtered["Revenue"].sum()
                total_profit = filtered["Profit"].sum()
                total_quantity = filtered["Quantity"].sum()

                total_orders = (
                    filtered["Order_ID"].nunique()
                    if "Order_ID" in filtered.columns
                    else len(filtered)
                )

                total_products = (
                    filtered["Product_Name"].nunique()
                    if "Product_Name" in filtered.columns
                    else 0
                )

                st.markdown("---")
                st.subheader("📈 Business Overview")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Total Revenue", money(total_revenue))
                c2.metric("Total Profit", money(total_profit))
                c3.metric("Total Orders", f"{total_orders:,}")
                c4.metric("Total Products", f"{total_products:,}")

                # ---------------- CHARTS ----------------

                st.markdown("---")

                                # ---------------- TOP SELLING PRODUCTS ----------------
                st.subheader("🏆 Top Selling Products")

                if "Product_Name" in filtered.columns:
                    top_products = (
                        filtered.groupby("Product_Name")["Quantity"]
                        .sum()
                        .nlargest(10)
                        .reset_index()
                    )

                    fig = px.bar(
                        top_products,
                        x="Product_Name",
                        y="Quantity",
                        title="Top 10 Products",
                        template=chart_template
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )
                else:
                    st.info("Product Name column is not available.")

                # ---------------- REVENUE BY CATEGORY ----------------
                st.subheader("💰 Revenue by Category")

                if "Category" in filtered.columns:
                    category_data = (
                        filtered.groupby("Category")["Revenue"]
                        .sum()
                        .reset_index()
                    )

                    fig = px.pie(
                        category_data,
                        names="Category",
                        values="Revenue",
                        hole=0.45,
                        template=chart_template
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )
                else:
                    st.info("Category column is not available.")

                                # ---------------- REVENUE VS PROFIT ----------------
                st.subheader("📊 Revenue vs Profit")

                comparison = pd.DataFrame({
                    "Metric": ["Revenue", "Profit"],
                    "Amount": [total_revenue, total_profit]
                })

                fig = px.bar(
                    comparison,
                    x="Metric",
                    y="Amount",
                    color="Metric",
                    template=chart_template,
                    text="Amount"
                )

                fig.update_traces(
                    texttemplate="%{text:,.0f}",
                    textposition="outside"
                )

                fig.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=30, r=30, t=70, b=40)
                )

                st.plotly_chart(fig, use_container_width=True)

                # ---------------- MONTHLY REVENUE TREND ----------------
                st.subheader("📅 Monthly Revenue Trend")

                if "Order_Date" in filtered.columns:
                    monthly = filtered.dropna(subset=["Order_Date"]).copy()

                    if not monthly.empty:
                        monthly["Month"] = (
                            monthly["Order_Date"].dt.to_period("M").astype(str)
                        )

                        monthly_data = (
                            monthly.groupby("Month")["Revenue"]
                            .sum()
                            .reset_index()
                        )

                        fig = px.line(
                            monthly_data,
                            x="Month",
                            y="Revenue",
                            markers=True,
                            template=chart_template
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No valid order dates found.")
                else:
                    st.info("Order Date column is not available.")

                st.markdown("---")
                st.subheader("📋 Filtered Sales Data")

                st.dataframe(
                    filtered,
                    use_container_width=True
                )

# =====================================================
# 11. SMART INVENTORY
# =====================================================

elif page == "Smart Inventory":

    st.title("📦 Smart Inventory Management")

    inventory = st.session_state.inventory_df

    if inventory is None:
        st.warning("Please upload Inventory Dataset first.")

    else:
        inventory = standardize_columns(inventory)
        inventory = convert_numeric(
            inventory,
            ["Available_Stock", "Reorder_Level"]
        )

        required = ["Available_Stock", "Reorder_Level"]
        missing = missing_columns(inventory, required)

        if missing:
            st.error("Missing Inventory Columns: " + ", ".join(missing))

        else:

            inventory["Stock_Status"] = np.select(
                [
                    inventory["Available_Stock"] <= 0,
                    inventory["Available_Stock"] <= inventory["Reorder_Level"],
                    inventory["Available_Stock"] > inventory["Reorder_Level"] * 3
                ],
                [
                    "Out of Stock",
                    "Low Stock",
                    "Overstock"
                ],
                default="Healthy Stock"
            )

            out_of_stock = inventory[
                inventory["Stock_Status"] == "Out of Stock"
            ]

            low_stock = inventory[
                inventory["Stock_Status"] == "Low Stock"
            ]

            healthy_stock = inventory[
                inventory["Stock_Status"] == "Healthy Stock"
            ]

            overstock = inventory[
                inventory["Stock_Status"] == "Overstock"
            ]

            st.subheader("📊 Inventory Overview")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Total Stock",
                f"{inventory['Available_Stock'].sum():,.0f}"
            )

            c2.metric("Out of Stock", len(out_of_stock))
            c3.metric("Low Stock", len(low_stock))
            c4.metric("Healthy Stock", len(healthy_stock))

            st.markdown("---")

            st.subheader("🚨 Inventory Alerts")

            if not out_of_stock.empty:
                st.error(
                    f"{len(out_of_stock)} products are out of stock!"
                )

                st.dataframe(
                    out_of_stock,
                    use_container_width=True
                )

            if not low_stock.empty:
                st.warning(
                    f"{len(low_stock)} products need restocking."
                )

                st.dataframe(
                    low_stock,
                    use_container_width=True
                )

            if out_of_stock.empty and low_stock.empty:
                st.success("All products have sufficient stock.")

            st.markdown("---")

            left, right = st.columns(2)

            with left:
                st.subheader("Stock Status Distribution")

                status_data = (
                    inventory["Stock_Status"]
                    .value_counts()
                    .reset_index()
                )

                status_data.columns = [
                    "Stock Status",
                    "Products"
                ]

                fig = px.pie(
                    status_data,
                    names="Stock Status",
                    values="Products",
                    hole=0.45,
                    template=chart_template
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with right:
                st.subheader("Stock by Product")

                if "Product_Name" in inventory.columns:
                    fig = px.bar(
                        inventory,
                        x="Product_Name",
                        y="Available_Stock",
                        color="Stock_Status",
                        template=chart_template
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )
                else:
                    st.info("Product Name column is not available.")

            st.markdown("---")

            st.subheader("💡 Reorder Suggestions")

            reorder_data = inventory[
                inventory["Available_Stock"] <= inventory["Reorder_Level"]
            ].copy()

            if not reorder_data.empty:
                reorder_data["Suggested_Reorder"] = (
                    reorder_data["Reorder_Level"]
                    - reorder_data["Available_Stock"]
                ).clip(lower=0)

                st.dataframe(
                    reorder_data,
                    use_container_width=True
                )
            else:
                st.success("No reorder suggestions at the moment.")

            st.markdown("---")
            st.subheader("📋 Complete Inventory Details")

            st.dataframe(
                inventory,
                use_container_width=True
            )

            csv_data = inventory.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Inventory CSV",
                data=csv_data,
                file_name="SmartStock_Inventory_Report.csv",
                mime="text/csv"
            )

# =====================================================
# 12. AI DEMAND FORECASTING
# =====================================================

elif page == "AI Demand Forecasting":

    st.title("🔮 AI Demand Forecasting")

    sales = st.session_state.sales_df

    if sales is None:
        st.warning("Please upload Sales Dataset first.")

    else:
        sales = standardize_columns(sales)

        required = ["Order_Date", "Quantity"]

        missing = missing_columns(sales, required)

        if missing:
            st.error(
                "Missing required columns: "
                + ", ".join(missing)
            )

        else:
            sales = convert_numeric(sales, ["Quantity"])

            sales["Order_Date"] = pd.to_datetime(
                sales["Order_Date"],
                errors="coerce"
            )

            sales = sales.dropna(
                subset=["Order_Date"]
            )

            if sales.empty:
                st.warning("No valid order dates found.")

            else:
                if "Product_Name" in sales.columns:
                    products = sorted(
                        sales["Product_Name"].dropna().astype(str).unique()
                    )

                    selected_product = st.selectbox(
                        "Select Product",
                        products
                    )

                    product_sales = sales[
                        sales["Product_Name"].astype(str) == selected_product
                    ].copy()

                else:
                    selected_product = "All Products"
                    product_sales = sales.copy()

                daily = (
                    product_sales.groupby("Order_Date")["Quantity"]
                    .sum()
                    .reset_index()
                )

                daily = daily.set_index("Order_Date").asfreq(
                    "D",
                    fill_value=0
                ).reset_index()

                if len(daily) < 14:
                    st.warning(
                        "At least 14 days of sales history is required "
                        "for demand forecasting."
                    )

                else:
                    daily["Day_Number"] = np.arange(len(daily))

                    X = daily[["Day_Number"]]
                    y = daily["Quantity"]

                    split = int(len(daily) * 0.8)

                    X_train = X.iloc[:split]
                    X_test = X.iloc[split:]
                    y_train = y.iloc[:split]
                    y_test = y.iloc[split:]

                    model = RandomForestRegressor(
                        n_estimators=100,
                        random_state=42
                    )

                    model.fit(X_train, y_train)

                    predictions = model.predict(X_test)

                    mae = mean_absolute_error(
                        y_test,
                        predictions
                    )

                    st.metric(
                        "Model Mean Absolute Error",
                        f"{mae:.2f} units"
                    )

                    model.fit(X, y)

                    future_days = 7

                    future_numbers = np.arange(
                        len(daily),
                        len(daily) + future_days
                    ).reshape(-1, 1)

                    forecast_values = model.predict(
                        future_numbers
                    )

                    future_dates = pd.date_range(
                        start=daily["Order_Date"].max()
                        + pd.Timedelta(days=1),
                        periods=future_days
                    )

                    forecast = pd.DataFrame({
                        "Order_Date": future_dates,
                        "Forecast_Quantity": np.maximum(
                            forecast_values, 0
                        ).round(0)
                    })

                    st.markdown("---")
                    st.subheader("📈 Demand Forecast")

                    history_chart = daily[
                        ["Order_Date", "Quantity"]
                    ].copy()

                    history_chart["Type"] = "Actual"

                    forecast_chart = forecast.rename(
                        columns={"Forecast_Quantity": "Quantity"}
                    )

                    forecast_chart["Type"] = "Forecast"

                    combined = pd.concat(
                        [history_chart, forecast_chart],
                        ignore_index=True
                    )

                    fig = px.line(
                        combined,
                        x="Order_Date",
                        y="Quantity",
                        color="Type",
                        markers=True,
                        template=chart_template,
                        title="Actual and Forecasted Demand"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.subheader("Next 7 Days Forecast")

                    st.dataframe(
                        forecast,
                        use_container_width=True
                    )

                    st.metric(
                        "Expected 7-Day Demand",
                        f"{forecast['Forecast_Quantity'].sum():,.0f} units"
                    )

                    forecast_csv = forecast.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "⬇️ Download Forecast CSV",
                        data=forecast_csv,
                        file_name="SmartStock_Demand_Forecast.csv",
                        mime="text/csv"
                    )

# =====================================================
# 13. AI BUSINESS ASSISTANT
# =====================================================

elif page == "AI Business Assistant":

    st.title("🤖 AI Business Assistant")

    st.write(
        "Ask questions about your uploaded business data."
    )

    st.info(
        "Examples: What is my total revenue? "
        "Which product sells the most? "
        "How many products have low stock?"
    )

    question = st.text_input(
        "Ask your business question"
    )

    if st.button("✨ Get Answer"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            answer = business_answer(
                question,
                st.session_state.sales_df,
                st.session_state.inventory_df
            )

            st.success(answer)

    st.caption(
        "The assistant uses rules to answer questions "
        "from your uploaded data."
    )

# =====================================================
# 14. REPORTS & DOWNLOADS
# =====================================================

elif page == "Reports & Downloads":

    st.title("📑 Reports & Downloads")

    sales = st.session_state.sales_df
    inventory = st.session_state.inventory_df

    if sales is None and inventory is None:
        st.warning("Please upload your datasets first.")

    else:

        if sales is not None:
            st.subheader("📊 Sales Report")

            st.dataframe(
                sales.head(10),
                use_container_width=True
            )

            sales_csv = sales.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Sales CSV",
                data=sales_csv,
                file_name="SmartStock_Sales_Report.csv",
                mime="text/csv"
            )

        if inventory is not None:
            st.subheader("📦 Inventory Report")

            st.dataframe(
                inventory.head(10),
                use_container_width=True
            )

            inventory_csv = inventory.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Inventory CSV",
                data=inventory_csv,
                file_name="SmartStock_Inventory_Report.csv",
                mime="text/csv"
            )

        st.markdown("---")

        excel_report = make_excel_report(
            sales,
            inventory
        )

        st.download_button(
            "📥 Download Complete Excel Report",
            data=excel_report,
            file_name="SmartStock_AI_Complete_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =====================================================
# 15. SIDEBAR FOOTER
# =====================================================

st.sidebar.markdown("---")
st.sidebar.caption("SmartStock AI")
st.sidebar.caption("Sales • Inventory • Analytics")
st.sidebar.caption("Built with Python & Streamlit")