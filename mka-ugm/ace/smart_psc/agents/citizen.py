from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message

class CitizenAgent(Agent):
    class RequestBehaviour(OneShotBehaviour):
        async def run(self):
            msg = Message(
                to="router@jabber.at",
                body="REQUEST:KTP",
                metadata={"performative": "request"}
            )
            await self.send(msg)
            print("[Citizen] Request sent")

    async def setup(self):
        self.add_behaviour(self.RequestBehaviour())
