
import streamlit as st
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any

st.set_page_config(page_title="MAS Smart Public Service (Demo)", layout="wide")

# -------------------- Data Structures --------------------
@dataclass
class Message:
    id: str
    sender: str
    receiver: str
    type: str
    content: Dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "id": self.id,
            "ts": time.strftime("%H:%M:%S", time.localtime(self.ts)),
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type,
            "content": str(self.content),
        }

class BaseAgent:
    def __init__(self, name):
        self.name = name
        self.state = "Idle"
        self.inbox: List[Message] = []
        self.log: List[str] = []

    def receive(self, msg: Message):
        self.inbox.append(msg)
        self.log.append(f"recv {msg.type} from {msg.sender}")

    def step(self, sim):
        """Override in subclass: process inbox / produce messages by calling sim.send(...)"""
        pass

    def info(self):
        return {"agent": self.name, "state": self.state, "inbox": len(self.inbox)}


# -------------------- Agent Implementations --------------------
class RouterAgent(BaseAgent):
    def step(self, sim):
        if not self.inbox:
            self.state = "Idle"
            return
        self.state = "Routing"
        while self.inbox:
            msg = self.inbox.pop(0)
            if msg.type == "REQUEST":
                # simple classification
                service = msg.content.get("service")
                target = "Dept_Dukcapil" if service == "KTP" else "Dept_Permit" if service == "Izin" else "Dept_Service"
                new_msg = Message(id=str(uuid.uuid4()), sender=self.name, receiver="NegotiatorAgent", type="ROUTE",
                                  content={"req_id": msg.content["req_id"], "service": service, "target": target, "priority": msg.content.get("priority",1)})
                sim.send(new_msg)
                self.log.append(f"Routed {msg.content['req_id']} -> {target}")

class NegotiatorAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)
        # simple policy: round-robin or shortest-queue can be implemented
        self.counter = 0

    def step(self, sim):
        if not self.inbox:
            self.state = "Idle"
            return
        self.state = "Negotiating"
        while self.inbox:
            msg = self.inbox.pop(0)
            if msg.type == "ROUTE":
                # propose to target department
                target = msg.content["target"]
                # create assignment proposal to the chosen department
                assign_msg = Message(id=str(uuid.uuid4()), sender=self.name, receiver=target, type="ASSIGN",
                                     content={"req_id": msg.content["req_id"], "service": msg.content["service"], "priority": msg.content["priority"]})
                sim.send(assign_msg)
                self.log.append(f"Proposed assign {msg.content['req_id']} -> {target}")

class DepartmentAgent(BaseAgent):
    def __init__(self, name, capacity=1, process_time=3):
        super().__init__(name)
        self.capacity = capacity
        self.process_time = process_time  # steps to complete a task
        self.tasks: List[Dict[str, Any]] = []  # each task: {"req_id", "remaining"}
    
    def step(self, sim):
        # Accept new tasks if capacity allows
        self.state = "Available" if len(self.tasks) < self.capacity else "Busy"
        # process inbox for ASSIGN messages
        while self.inbox:
            msg = self.inbox.pop(0)
            if msg.type == "ASSIGN":
                if len(self.tasks) < self.capacity:
                    self.tasks.append({"req_id": msg.content["req_id"], "remaining": self.process_time, "priority": msg.content.get("priority",1), "source": msg.sender})
                    self.log.append(f"Accepted {msg.content['req_id']}")
                    # inform negotiator/sender that accepted
                    inform = Message(id=str(uuid.uuid4()), sender=self.name, receiver=msg.sender, type="ACCEPT",
                                     content={"req_id": msg.content["req_id"]})
                    sim.send(inform)
                else:
                    # reject due to overload
                    self.log.append(f"Rejected {msg.content['req_id']} (overload)")
                    reject = Message(id=str(uuid.uuid4()), sender=self.name, receiver=msg.sender, type="REJECT",
                                     content={"req_id": msg.content["req_id"]})
                    sim.send(reject)
        # process active tasks
        for t in list(self.tasks):
            t["remaining"] -= 1
            if t["remaining"] <= 0:
                self.tasks.remove(t)
                done = Message(id=str(uuid.uuid4()), sender=self.name, receiver="SupervisorAgent", type="DONE",
                               content={"req_id": t["req_id"], "dept": self.name})
                sim.send(done)
                self.log.append(f"Completed {t['req_id']}")
        # update state
        self.state = "Available" if len(self.tasks) < self.capacity else "Overloaded"

class SupervisorAgent(BaseAgent):
    def step(self, sim):
        # monitor DONE messages or REJECT and perform recovery
        if not self.inbox:
            self.state = "Monitoring"
            return
        while self.inbox:
            msg = self.inbox.pop(0)
            if msg.type == "DONE":
                # record completion
                sim.metrics["completed"] += 1
                sim.logs.append(f"[Supervisor] Completed {msg.content['req_id']} at {msg.content['dept']}")
            elif msg.type == "REJECT":
                # reassign: send back to Negotiator for re-negotiation
                sim.logs.append(f"[Supervisor] Reassigning {msg.content['req_id']} due to rejection")
                re_route = Message(id=str(uuid.uuid4()), sender=self.name, receiver="NegotiatorAgent", type="REASSIGN",
                                   content={"req_id": msg.content["req_id"], "origin": msg.sender})
                sim.send(re_route)
        self.state = "Monitoring"

class CitizenAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)
        self.requests: List[Dict[str, Any]] = []

    def create_request(self, service, priority=1):
        req_id = str(uuid.uuid4())[:8]
        req = {"req_id": req_id, "service": service, "priority": priority, "citizen": self.name, "ts": time.time()}
        self.requests.append(req)
        return req

    def step(self, sim):
        # citizens are passive in our simple sim; they create requests via UI
        # optionally they can resend if not completed; for simplicity: do nothing here
        pass

# -------------------- Simulation Controller --------------------
class Simulation:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: List[Message] = []
        self.logs: List[str] = []
        self.metrics = {"created":0, "completed":0}
    
    def send(self, msg: Message):
        # append to queue (will be delivered after all agents step)
        self.message_queue.append(msg)
        # also append to global logs
        self.logs.append(f"{time.strftime('%H:%M:%S')} | {msg.sender} -> {msg.receiver} : {msg.type} | {msg.content.get('req_id','')}")

    def deliver_messages(self):
        # deliver all messages to recipients' inboxes
        queue = list(self.message_queue)
        self.message_queue = []
        for msg in queue:
            recipient = self.agents.get(msg.receiver)
            if recipient:
                recipient.receive(msg)
            else:
                # if no agent, log dropped message
                self.logs.append(f"{time.strftime('%H:%M:%S')} | DROPPED message to {msg.receiver}: {msg.type}")

    def step(self):
        # step each agent (in deterministic order)
        for name, agent in list(self.agents.items()):
            agent.step(self)
        # deliver messages after all agents processed their logic
        self.deliver_messages()


# -------------------- Streamlit App --------------------
def init_sim():
    if "sim" not in st.session_state:
        sim = Simulation()
        # create agents
        sim.agents["CitizenAgent"] = CitizenAgent("CitizenAgent")
        sim.agents["RouterAgent"] = RouterAgent("RouterAgent")
        sim.agents["NegotiatorAgent"] = NegotiatorAgent("NegotiatorAgent")
        sim.agents["Dept_Dukcapil"] = DepartmentAgent("Dept_Dukcapil", capacity=1, process_time=3)
        sim.agents["Dept_Permit"] = DepartmentAgent("Dept_Permit", capacity=1, process_time=4)
        sim.agents["Dept_Service"] = DepartmentAgent("Dept_Service", capacity=2, process_time=2)
        sim.agents["SupervisorAgent"] = SupervisorAgent("SupervisorAgent")
        st.session_state.sim = sim
        st.session_state.message_log = []
    return st.session_state.sim

sim = init_sim()

st.title("Multi-Agent System — Smart Public Service (Streamlit Demo)")
st.markdown("Demo sederhana untuk tugas kuliah: setiap tombol `Step` menjalankan satu siklus agen. Sistem menggunakan message passing sederhana dan agent state machines.")

# Left column: input & controls
col1, col2 = st.columns([1,2])

with col1:
    st.header("Create Request")
    citizen_name = st.text_input("Citizen name", value="Tomy")
    service = st.selectbox("Service", ["KTP", "Izin", "Pengaduan"])
    priority = st.slider("Priority (1=normal, higher=urgent)", 1, 5, 1)
    if st.button("Submit Request"):
        citizen = sim.agents["CitizenAgent"]
        req = citizen.create_request(service, priority)
        sim.metrics["created"] += 1
        # send to router as initial REQUEST message
        msg = Message(id=str(uuid.uuid4()), sender="CitizenAgent", receiver="RouterAgent", type="REQUEST", content=req)
        sim.send(msg)
        st.success(f"Submitted request {req['req_id']} for {service} (priority {priority})")
    st.markdown("---")
    if st.button("Step (1)"):
        sim.step()
    if st.button("Run 5 steps"):
        for _ in range(5):
            sim.step()
    if st.button("Reset Simulation"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

    st.markdown("**Metrics**")
    st.write(sim.metrics)

with col2:
    st.header("Agents Overview")
    # show agents table
    ag_info = [agent.info() for agent in sim.agents.values()]
    import pandas as pd
    df = pd.DataFrame(ag_info)
    # enrich with tasks if department
    def tasks_for(agent):
        if isinstance(agent, DepartmentAgent):
            return ", ".join([t["req_id"] for t in agent.tasks]) or "-"
        return "-"
    df["tasks"] = [tasks_for(sim.agents[name]) for name in df["agent"]]
    df = df[["agent","state","inbox","tasks"]]
    st.dataframe(df, use_container_width=True)

    st.markdown("### Message Log (recent)")
    log_display = sim.logs[-100:][::-1]
    for l in log_display[:200]:
        st.text(l)

# Bottom: agent internal logs
st.markdown("---")
st.header("Agent Internal Logs (debug)")
selected_agent = st.selectbox("Select agent to view logs", list(sim.agents.keys()))
if selected_agent:
    agent = sim.agents[selected_agent]
    st.write(f"State: {agent.state} | Inbox: {len(agent.inbox)}")
    for i, line in enumerate(agent.log[::-1][:100]):
        st.text(f"{i+1}. {line}")

st.markdown("---")
st.caption("Design note: This is a simplified MAS simulation suitable for classroom demonstration. It shows autonomy (agents act on their own per step), communication (message passing), negotiation (Negotiator proposes assignments), and failure handling (Supervisor reassigns on REJECT).")
