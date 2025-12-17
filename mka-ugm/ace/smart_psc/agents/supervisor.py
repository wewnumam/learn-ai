from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

class SupervisorAgent(Agent):
    class MonitorBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print("[Supervisor] Service completed")

    async def setup(self):
        self.add_behaviour(self.MonitorBehaviour())
