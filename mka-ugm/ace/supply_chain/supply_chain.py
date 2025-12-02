import random
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ================================
#          AGENT MODEL
# ================================
class Agent:
    def __init__(self, name, upstream=None, lead_time=2, init_inventory=20, target_inventory=30):
        self.name = name
        self.upstream = upstream
        self.lead_time = lead_time

        self.inventory = init_inventory
        self.backlog = 0

        self.shipments = [0] * lead_time
        self.incoming_orders = [0] * lead_time
        self.current_order = 0

        self.target_inventory = target_inventory
        self.last_order_placed = 0

    def incoming_shipments_sum(self):
        return sum(self.shipments)

    def incoming_orders_sum(self):
        return sum(self.incoming_orders)


# ================================
#      SUPPLY CHAIN SIMULATOR
# ================================
class SupplyChainSim:
    def __init__(self, lead_time=2, seed=1, targets=None):
        random.seed(seed)

        # Use user-defined targets
        t_retailer, t_wholesaler, t_distributor = targets

        self.manufacturer = Agent("Manufacturer", upstream=None, lead_time=lead_time,
                                  init_inventory=9999999, target_inventory=0)
        self.distributor = Agent("Distributor", upstream=self.manufacturer, lead_time=lead_time,
                                 init_inventory=25, target_inventory=t_distributor)
        self.wholesaler = Agent("Wholesaler", upstream=self.distributor, lead_time=lead_time,
                                init_inventory=20, target_inventory=t_wholesaler)
        self.retailer = Agent("Retailer", upstream=self.wholesaler, lead_time=lead_time,
                              init_inventory=15, target_inventory=t_retailer)

        self.agents = [self.retailer, self.wholesaler, self.distributor, self.manufacturer]
        self.period = 0

    def step(self, external_demand):
        self.period += 1

        # Step 1: Advance pipelines
        for a in self.agents:
            arriving_ship = a.shipments.pop(0)
            a.shipments.append(0)

            if a.upstream is not None:
                a.inventory += arriving_ship

            a.current_order = a.incoming_orders.pop(0)
            a.incoming_orders.append(0)

        self.retailer.current_order += external_demand

        # Step 2: Fulfillment
        for i, a in enumerate(self.agents):
            total_demand = a.backlog + a.current_order

            if a.upstream is None:
                shipped = total_demand
                a.backlog = 0
            else:
                shipped = min(a.inventory, total_demand)
                a.inventory -= shipped
                a.backlog = total_demand - shipped

            if i > 0:
                downstream = self.agents[i - 1]
                downstream.shipments[-1] += shipped

        # Step 3: Ordering
        for a in self.agents:
            if a.upstream is None:
                a.last_order_placed = 0
                continue

            net_inventory = a.inventory + a.incoming_shipments_sum() - a.backlog
            order = max(0, a.target_inventory - net_inventory)
            order = int(order)

            a.last_order_placed = order
            a.upstream.incoming_orders[-1] += order

        # Prepare snapshot
        snapshot = {
            "period": self.period,
            "external_demand": external_demand,
            "agents": [
                {
                    "Name": a.name,
                    "Inventory": a.inventory if a.inventory < 999999 else float("inf"),
                    "Backlog": a.backlog,
                    "Incoming Shipments": a.incoming_shipments_sum(),
                    "Incoming Orders": a.incoming_orders_sum(),
                    "Last Order Placed": a.last_order_placed,
                }
                for a in self.agents
            ]
        }
        return snapshot


# ================================
#        STREAMLIT UI
# ================================
st.title("📦 Supply Chain Simulation (Retailer → Wholesaler → Distributor → Manufacturer)")
st.write("Simulasi model Beer Game sederhana dengan Streamlit.")

st.sidebar.header("⚙️ Simulation Settings")
lead_time = st.sidebar.slider("Lead Time (period)", 1, 5, 2)
periods = st.sidebar.slider("Number of Periods", 5, 50, 20)
seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

st.sidebar.header("🎯 Target Inventory (Order-Up-To Levels)")
t_retailer = st.sidebar.slider("Retailer Target Inventory", 0, 100, 30)
t_wholesaler = st.sidebar.slider("Wholesaler Target Inventory", 0, 100, 30)
t_distributor = st.sidebar.slider("Distributor Target Inventory", 0, 100, 30)

run_button = st.button("▶ Run Simulation")

# ================================
#       SIMULATION EXECUTION
# ================================
if run_button:
    sim = SupplyChainSim(
        lead_time=lead_time,
        seed=seed,
        targets=(t_retailer, t_wholesaler, t_distributor)
    )

    logs = []
    df_records = []

    st.subheader("📘 Simulation Log")

    for t in range(periods):
        demand = random.randint(2, 6)
        snap = sim.step(demand)

        # For terminal-like log
        log_text = f"Period {snap['period']:2d} | Demand {snap['external_demand']}\n"
        for info in snap["agents"]:
            inv = info["Inventory"]
            inv_str = "inf" if inv == float("inf") else str(inv)
            log_text += (
                f"  {info['Name'][:10].ljust(10)} "
                f"inv={inv_str:>4} "
                f"back={info['Backlog']:>3} "
                f"in_ship={info['Incoming Shipments']:>3} "
                f"in_ord={info['Incoming Orders']:>3} "
                f"last_ord={info['Last Order Placed']:>3}\n"
            )
        log_text += "-" * 72
        logs.append(log_text)

        # For DataFrame
        for info in snap["agents"]:
            df_records.append({
                "Period": snap["period"],
                "Demand": snap["external_demand"],
                **info
            })

    # Display log
    st.code("\n\n".join(logs), language="text")

    # Display DataFrame
    st.subheader("📊 Simulation Table")
    df = pd.DataFrame(df_records)
    st.dataframe(df, use_container_width=True)

    # ================================
    #      TIME SERIES VISUALS
    # ================================
    # Clean up infinite inventory values (manufacturer)
    df_plot = df.copy()
    df_plot["Inventory"] = df_plot["Inventory"].replace([float("inf"), np.inf], np.nan).astype(float)

    # Allow larger datasets for Altair if needed
    alt.data_transformers.disable_max_rows()

    st.subheader("📈 Time Series")

    # Helper to build a chart for a given metric
    def make_line_chart(metric, y_label=None):
        chart = (
            alt.Chart(df_plot)
            .mark_line(point=True)
            .encode(
                x=alt.X("Period:Q"),
                y=alt.Y(f"{metric}:Q", title=y_label or metric),
                color=alt.Color("Name:N", title="Agent"),
                tooltip=["Period", "Name", metric]
            )
            .interactive()
            .properties(height=250)
        )
        return chart

    # Inventory chart
    st.caption("Inventory over time (manufacturer inf values hidden)")
    st.altair_chart(make_line_chart("Inventory", "Inventory"), use_container_width=True)

    # Backlog chart
    st.caption("Backlog over time")
    st.altair_chart(make_line_chart("Backlog", "Backlog"), use_container_width=True)

    # Orders chart
    st.caption("Last Order Placed over time")
    st.altair_chart(make_line_chart("Last Order Placed", "Order Placed"), use_container_width=True)

    st.success("Simulation Completed!")
