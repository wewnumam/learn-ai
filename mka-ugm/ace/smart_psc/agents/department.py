from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

class DepartmentAgent(Agent):
    class ProcessBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print("[Department] Processing service")
                reply = Message(
                    to="supervisor@jabber.at",
                    body="SERVICE:DONE",
                    metadata={"performative": "inform"}
                )
                await self.send(reply)

    async def setup(self):
        self.add_behaviour(self.ProcessBehaviour())
