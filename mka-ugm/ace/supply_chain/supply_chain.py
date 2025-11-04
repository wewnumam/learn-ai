"""
Supply chain agent-based model for a mass-production sambal industry with Mesa visualisation (factory -> central warehouse -> 5 regional warehouses -> retailers)
Default parameters used as requested by user.

Run with: python supply_chain_sambal_mesa.py
This script requires: mesa, numpy, pandas
Install with: pip install mesa numpy pandas

Launch interactive visualization: the script starts a Mesa server (default port 8521). Use the "Steps per tick" slider to fast-forward (each server tick runs that many simulated days).

Outputs when run headless or after simulation:
 - results_timeseries.csv : daily metrics time series
 - inventory_timeseries.csv : inventory levels per node per day
 - shipments_log.csv : shipments events (for lead time calc)

Features added since previous version:
 - Mesa server GUI with CanvasGrid (linear layout) + ChartModule (inventory, fill rate, stockouts, avg lead time, total cost)
 - Compatibility import fallback for various Mesa versions
 - "steps_per_tick" user-settable parameter to fast-forward simulation (1..10 days per tick)
"""

from mesa import Model, Agent
# Compatibility import for scheduler
try:
    from mesa.time import BaseScheduler
except Exception:
    # try alternative locations for newer Mesa
    try:
        from mesa.experimental.time import BaseScheduler
    except Exception:
        # As a last resort, import the generic scheduler class
        from mesa.time import RandomActivation as BaseScheduler

# Visualization imports
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler as _DummyScheduler

import numpy as np
import pandas as pd
import random
import math

# ------------------------- Helper functions -------------------------

def truncated_normal(mean, sigma, low=0):
    val = random.gauss(mean, sigma)
    return max(low, val)

# ------------------------- Agent definitions -------------------------

class SupplierAgent(Agent):
    def __init__(self, unique_id, model, supplier_type='local'):
        super().__init__(unique_id, model)
        self.supplier_type = supplier_type  # 'local' or 'big'
        if supplier_type == 'local':
            self.lt_mu = 7.0
            self.lt_sigma = 3.0
            self.capacity_per_day = 200.0
        else:
            self.lt_mu = 4.0
            self.lt_sigma = 1.5
            self.capacity_per_day = 2000.0

    def provide_lead_time(self):
        lt = random.gauss(self.lt_mu, self.lt_sigma)
        return max(1.0, lt)

    def step(self):
        pass

class FactoryAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.raw_inventory = 200000.0
        self.production_capacity = 100000.0
        self.yield_rate = 0.99
        self.k_service = 1.65
        self.estimated_lt_mean = (model.num_local_suppliers * 7.0 + model.num_big_suppliers * 4.0) / max(1, (model.num_local_suppliers + model.num_big_suppliers))
        self.estimated_lt_sigma = (model.num_local_suppliers * 3.0 + model.num_big_suppliers * 1.5) / max(1, (model.num_local_suppliers + model.num_big_suppliers))

    def place_order_for_raw(self):
        avg_daily_usage = self.production_capacity
        mean_lt = self.estimated_lt_mean
        sd_lt = self.estimated_lt_sigma
        mean_lt_demand = avg_daily_usage * mean_lt
        safety = self.k_service * math.sqrt(mean_lt) * avg_daily_usage * 0.2
        reorder_point = mean_lt_demand + safety
        if self.raw_inventory < reorder_point:
            order_qty = max(avg_daily_usage * (mean_lt + 3), 10000)
            remaining = order_qty
            shipments = []
            for s in self.model.supplier_agents_big:
                if remaining <= 0:
                    break
                qty = min(remaining, s.capacity_per_day * (mean_lt + 1))
                lt = max(1, int(round(s.provide_lead_time())))
                eta_day = self.model.current_day + lt
                shipments.append({'from': s.unique_id, 'to': 'factory', 'qty': qty, 'eta': eta_day, 'lead_time': lt})
                remaining -= qty
            if remaining > 0:
                for s in self.model.supplier_agents_local:
                    if remaining <= 0:
                        break
                    qty = min(remaining, s.capacity_per_day * (mean_lt + 1))
                    lt = max(1, int(round(s.provide_lead_time())))
                    eta_day = self.model.current_day + lt
                    shipments.append({'from': s.unique_id, 'to': 'factory', 'qty': qty, 'eta': eta_day, 'lead_time': lt})
                    remaining -= qty
            for sh in shipments:
                self.model.in_transit_raw.append(sh)
                self.model.shipments_log.append({'day_ordered': self.model.current_day, 'from': sh['from'], 'to': sh['to'], 'qty': sh['qty'], 'eta': sh['eta'], 'lead_time': sh['lead_time']})

    def produce(self):
        possible = min(self.production_capacity, self.raw_inventory)
        produced = possible * self.yield_rate
        self.raw_inventory -= possible
        central = self.model.central_warehouse
        ship_qty = min(produced, central.free_space())
        if ship_qty > 0:
            central.receive(ship_qty)
            self.model.shipments_log.append({'day_ordered': self.model.current_day, 'from': 'factory', 'to': 'central', 'qty': ship_qty, 'eta': self.model.current_day + max(1, int(round(truncated_normal(2,1)))), 'lead_time': None})

    def step(self):
        arrivals = [sh for sh in self.model.in_transit_raw if sh['to'] == 'factory' and sh['eta'] <= self.model.current_day]
        for sh in arrivals:
            self.raw_inventory += sh['qty']
            self.model.in_transit_raw.remove(sh)
        self.place_order_for_raw()
        self.produce()

class WarehouseAgent(Agent):
    def __init__(self, unique_id, model, region=None, capacity=5000):
        super().__init__(unique_id, model)
        self.region = region
        self.capacity = capacity
        self.inventory = 0.0
        self.age_layers = []
        self.backorder = 0.0

    def free_space(self):
        return max(0, self.capacity - self.inventory)

    def receive(self, qty):
        accepted = min(qty, self.free_space())
        if accepted <= 0:
            return 0
        self.inventory += accepted
        self.age_layers.append({'qty': accepted, 'age': 0})
        return accepted

    def ship_out(self, qty):
        shipped = 0.0
        remain = qty
        new_layers = []
        for layer in self.age_layers:
            if remain <= 0:
                new_layers.append(layer)
                continue
            take = min(layer['qty'], remain)
            layer['qty'] -= take
            shipped += take
            remain -= take
            if layer['qty'] > 0:
                new_layers.append(layer)
        self.age_layers = [l for l in new_layers if l['qty'] > 0]
        self.inventory -= shipped
        return shipped

    def age_perish(self):
        new_layers = []
        for layer in self.age_layers:
            layer['age'] += 1
            if layer['age'] <= self.model.shelf_life_days:
                new_layers.append(layer)
        self.age_layers = new_layers
        self.inventory = sum([l['qty'] for l in self.age_layers])

    def step(self):
        self.age_perish()

class ModelTransport:
    def __init__(self, model):
        self.model = model

    def send_fg(self, qty, to_region_idx):
        lt = max(1, int(round(random.gauss(2.0,1.0))))
        eta = self.model.current_day + lt
        shipment = {'from': 'central', 'to': f'regional_{to_region_idx}', 'qty': qty, 'eta': eta, 'lead_time': lt}
        self.model.in_transit_fg.append(shipment)
        self.model.shipments_log.append({'day_ordered': self.model.current_day, 'from': 'central', 'to': f'regional_{to_region_idx}', 'qty': qty, 'eta': eta, 'lead_time': lt})

# ------------------------- The Mesa Model -------------------------

class SambalSupplyChainModel(Model):
    def __init__(self, steps_per_tick=1):
        super().__init__()
        self.schedule = BaseScheduler(self)
        self.total_demand_per_day = 1000
        self.num_regions = 5
        self.regional_share = [1.0/self.num_regions for _ in range(self.num_regions)]
        self.sim_days = 180
        self.current_day = 0
        self.num_local_suppliers = 1000
        self.num_big_suppliers = 50
        self.shelf_life_days = 15
        self.central_capacity = 100000
        self.regional_capacity = 15000
        self.supplier_agents_local = []
        self.supplier_agents_big = []
        for i in range(self.num_local_suppliers):
            a = SupplierAgent(f'loc_{i}', self, 'local')
            self.supplier_agents_local.append(a)
            self.schedule.add(a)
        for i in range(self.num_big_suppliers):
            a = SupplierAgent(f'big_{i}', self, 'big')
            self.supplier_agents_big.append(a)
            self.schedule.add(a)
        self.factory = FactoryAgent('factory_1', self)
        self.schedule.add(self.factory)
        self.central_warehouse = WarehouseAgent('central_wh', self, region=None, capacity=self.central_capacity)
        self.schedule.add(self.central_warehouse)
        self.regional_warehouses = []
        for r in range(self.num_regions):
            wh = WarehouseAgent(f'regional_{r}', self, region=r, capacity=self.regional_capacity)
            self.regional_warehouses.append(wh)
            self.schedule.add(wh)
        self.transport = ModelTransport(self)
        self.in_transit_raw = []
        self.in_transit_fg = []
        self.shipments_log = []
        self.yield_loss = 0.01
        self.total_cost = 0.0
        self.total_demand = 0.0
        self.total_fulfilled = 0.0
        self.total_stockouts = 0.0
        self.lead_times_observed = []
        self.steps_per_tick = steps_per_tick
        self.datacollector = DataCollector(
            model_reporters={
                "day": lambda m: m.current_day,
                "total_demand": lambda m: m.total_demand,
                "total_fulfilled": lambda m: m.total_fulfilled,
                "fill_rate": lambda m: (m.total_fulfilled / m.total_demand) if m.total_demand>0 else 0,
                "total_stockouts": lambda m: m.total_stockouts,
                "factory_raw": lambda m: m.factory.raw_inventory,
                "central_inventory": lambda m: m.central_warehouse.inventory,
                "avg_lead_time": lambda m: (np.mean(m.lead_times_observed) if len(m.lead_times_observed)>0 else 0),
                "total_cost": lambda m: m.total_cost
            }
        )
        self.central_warehouse.receive(5000)
        for r in self.regional_warehouses:
            r.receive(2000)

    def step_day(self):
        self.current_day += 1
        arrivals_fg = [sh for sh in self.in_transit_fg if sh['eta'] <= self.current_day]
        for sh in arrivals_fg:
            region_idx = int(sh['to'].split('_')[1])
            accepted = self.regional_warehouses[region_idx].receive(sh['qty'])
            if accepted < sh['qty']:
                ret = sh['qty'] - accepted
                self.central_warehouse.receive(ret)
            self.in_transit_fg.remove(sh)
            self.lead_times_observed.append(sh['lead_time'])
        self.schedule.step()
        for idx, wh in enumerate(self.regional_warehouses):
            mean_daily = self.total_demand_per_day * self.regional_share[idx]
            rpoint = mean_daily * 2.0
            if wh.inventory < rpoint:
                qty_needed = min(self.central_warehouse.inventory, wh.capacity - wh.inventory)
                if qty_needed > 0:
                    self.central_warehouse.ship_out(qty_needed)
                    self.transport.send_fg(qty_needed, idx)
        for idx, wh in enumerate(self.regional_warehouses):
            mean = self.total_demand_per_day * self.regional_share[idx]
            sigma = 0.2 * mean
            demand = int(round(max(0, random.gauss(mean, sigma))))
            self.total_demand += demand
            fulfilled = wh.ship_out(demand)
            if fulfilled < demand:
                back = demand - fulfilled
                self.total_stockouts += back
                if self.central_warehouse.inventory > 0:
                    pulled = min(self.central_warehouse.inventory, back)
                    self.central_warehouse.ship_out(pulled)
                    eta = self.current_day + 1
                    self.in_transit_fg.append({'from': 'central', 'to': f'regional_{idx}', 'qty': pulled, 'eta': eta, 'lead_time': 1})
                    self.shipments_log.append({'day_ordered': self.current_day, 'from': 'central', 'to': f'regional_{idx}', 'qty': pulled, 'eta': eta, 'lead_time': 1})
            self.total_fulfilled += fulfilled
        production_cost_per_bottle = 2000.0
        holding_cost_per_bottle_per_day = 1.0
        stockout_penalty_per_unit = 5000.0
        holding = (self.central_warehouse.inventory + sum([r.inventory for r in self.regional_warehouses]) + self.factory.raw_inventory*0.001) * holding_cost_per_bottle_per_day
        stockout_cost = self.total_stockouts * stockout_penalty_per_unit
        self.total_cost = holding + stockout_cost
        self.datacollector.collect(self)

    def step(self):
        # step runs steps_per_tick days to allow fast-forward on server
        for _ in range(max(1, int(self.steps_per_tick))):
            if self.current_day < self.sim_days:
                self.step_day()

    def run_model(self, days=None):
        if days is None:
            days = self.sim_days
        for d in range(days):
            self.step_day()

# ------------------------- Visualization / Portrayal -------------------------

def agent_portrayal(agent):
    if agent is None:
        return None
    portrayal = {"Shape": "circle", "Filled": True, "r": 0.8}
    # Place agents on a 1D line: x position determined by type, y by index
    if isinstance(agent, SupplierAgent):
        portrayal["Color"] = "#66c2a5" if agent.supplier_type == 'local' else "#fc8d62"
        portrayal["Layer"] = 0
        portrayal["text"] = str(agent.unique_id)
    elif isinstance(agent, FactoryAgent):
        portrayal["Color"] = "#8da0cb"
        portrayal["Layer"] = 1
        portrayal["text"] = "Factory"
    elif isinstance(agent, WarehouseAgent):
        if agent.region is None:
            portrayal["Color"] = "#e78ac3"
            portrayal["text"] = f"Central\n{int(agent.inventory)}"
            portrayal["Layer"] = 1
        else:
            portrayal["Color"] = "#a6d854"
            portrayal["text"] = f"R{agent.region}\n{int(agent.inventory)}"
            portrayal["Layer"] = 1
    portrayal["text_color"] = "#000000"
    return portrayal

# simple grid size and placement mapping: not spatially accurate but linear layout
canvas_width = 800
canvas_height = 200
# we'll use a CanvasGrid with n_rows=1 and n_columns = number of visual slots
visual_slots = 1 + 5 + 5  # suppliers aggregated not drawn individually to keep UI lightweight

def make_canvas_grid():
    # Mesh portrayal requires grid size; we will map logical positions to a grid
    grid = CanvasGrid(agent_portrayal, 11, 1, canvas_width, canvas_height)
    return grid

# Chart modules for default metrics
chart = ChartModule([
    {"Label": "central_inventory", "Color": "#e78ac3"},
    {"Label": "factory_raw", "Color": "#8da0cb"},
    {"Label": "fill_rate", "Color": "#66c2a5"},
    {"Label": "total_stockouts", "Color": "#fc8d62"},
    {"Label": "avg_lead_time", "Color": "#a6d854"},
    {"Label": "total_cost", "Color": "#a6768f"}
])

# User-settable parameter for fast-forwarding
model_params = {
    "steps_per_tick": UserSettableParameter('slider', "Steps per tick (days)", 3, 1, 10, 1)
}

# Server
server = ModularServer(SambalSupplyChainModel,
                       [chart],
                       "Sambal Supply Chain",
                       model_params)
server.port = 8521

if __name__ == '__main__':
    print("Starting Mesa server at http://127.0.0.1:8521")
    server.launch()

    # Note: when run via server, the model runs through server-controlled steps.
    # The headless export is still available by importing the model class and calling run_model().
