from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

class NegotiatorAgent(Agent):
    class NegotiateBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print("[Negotiator] Negotiating slot")
                reply = Message(
                    to="department@jabber.at",
                    body="SLOT:AVAILABLE",
                    metadata={"performative": "propose"}
                )
                await self.send(reply)

    async def setup(self):
        self.add_behaviour(self.NegotiateBehaviour())
