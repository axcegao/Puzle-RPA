"""db.py - загрузка и объединение данных из CSV-файлов."""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "Sk students data", "data")

TABLES = ["clients", "contracts", "outlets", "products",
          "regions", "discounts", "sales", "sales_plan"]

NA_VALUES = ["\\N"]

_cache = {}


def _read(name: str) -> pd.DataFrame:
    if name not in _cache:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        _cache[name] = pd.read_csv(path, na_values=NA_VALUES)
    return _cache[name]


def init():
    """Предзагрузка всех таблиц."""
    for name in TABLES:
        _read(name)


def get_sales_flat() -> pd.DataFrame:
    """Продажи в зерне sale_id со всеми измерениями."""
    sales = _read("sales").copy()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
    sales["period"] = sales["sale_date"].dt.to_period("M").dt.to_timestamp()

    outlets = _read("outlets")[["outlet_id", "outlet_name", "region_id", "city", "outlet_type"]]
    regions = _read("regions")[["region_id", "region_name"]]
    clients = _read("clients")[["client_id", "client_name", "segment"]]
    products = _read("products")[["product_id", "product_name", "category"]]
    contracts = _read("contracts")[["contract_id", "contract_number", "status"]]

    df = sales.merge(outlets, on="outlet_id", how="left")
    df = df.merge(regions, on="region_id", how="left")
    df = df.merge(clients, on="client_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(contracts, on="contract_id", how="left")
    df.rename(columns={"amount": "fact_amount", "quantity": "fact_qty"}, inplace=True)
    return df


def get_sales_with_plan() -> pd.DataFrame:
    """Плоская таблица в зерне месяц x точка x продукт: факт, план, измерения."""
    fact = get_sales_flat()
    fact = fact.groupby(["period", "outlet_id", "product_id"], as_index=False).agg(
        fact_amount=("fact_amount", "sum"),
        fact_qty=("fact_qty", "sum"),
        region_id=("region_id", "first"),
        region_name=("region_name", "first"),
        outlet_name=("outlet_name", "first"),
        city=("city", "first"),
        outlet_type=("outlet_type", "first"),
        product_name=("product_name", "first"),
        category=("category", "first"),
    )

    plan = _read("sales_plan").copy()
    plan["period"] = pd.to_datetime(plan["plan_month"], errors="coerce")
    plan = plan[["period", "outlet_id", "product_id", "plan_amount", "plan_qty"]]

    df = fact.merge(plan, on=["period", "outlet_id", "product_id"], how="outer")

    outlets = _read("outlets")[["outlet_id", "outlet_name", "region_id", "city", "outlet_type"]]
    regions = _read("regions")[["region_id", "region_name"]]
    products = _read("products")[["product_id", "product_name", "category"]]
    pairs = df[["outlet_id", "product_id"]].drop_duplicates()
    dims = pairs.merge(outlets, on="outlet_id", how="left")
    dims = dims.merge(regions, on="region_id", how="left")
    dims = dims.merge(products, on="product_id", how="left")
    dims = dims[["outlet_id", "product_id", "region_id", "region_name",
                 "outlet_name", "city", "outlet_type", "product_name", "category"]]
    df = df.merge(dims, on=["outlet_id", "product_id"], how="left", suffixes=("", "_ref"))
    ref_cols = [c for c in df.columns if c.endswith("_ref")]
    for col in ref_cols:
        df[col[:-4]] = df[col[:-4]].fillna(df[col])
    df.drop(columns=ref_cols, inplace=True)
    return df


def get_discounts() -> pd.DataFrame:
    """Скидки с измерениями: клиент, точка, регион, продукт."""
    disc = _read("discounts").copy()
    disc["discount_date"] = pd.to_datetime(disc["discount_date"], errors="coerce")
    disc["period"] = disc["discount_date"].dt.to_period("M").dt.to_timestamp()

    clients = _read("clients")[["client_id", "client_name", "segment"]]
    sales = _read("sales")[["sale_id", "sale_date", "outlet_id", "product_id", "amount"]].copy()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
    sales.rename(columns={"amount": "sale_amount"}, inplace=True)

    outlets = _read("outlets")[["outlet_id", "outlet_name", "region_id"]]
    regions = _read("regions")[["region_id", "region_name"]]
    products = _read("products")[["product_id", "product_name", "category"]]

    df = disc.merge(clients, on="client_id", how="left")
    df = df.merge(sales, on="sale_id", how="left")
    df = df.merge(outlets, on="outlet_id", how="left")
    df = df.merge(regions, on="region_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    return df


def get_clients() -> pd.DataFrame:
    return _read("clients").copy()


def get_contracts() -> pd.DataFrame:
    return _read("contracts").copy()


def get_outlets() -> pd.DataFrame:
    return _read("outlets").copy()


def get_regions() -> pd.DataFrame:
    return _read("regions").copy()


def get_products() -> pd.DataFrame:
    return _read("products").copy()