from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

class RouterAgent(Agent):
    class RouteBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print("[Router] Routing request")
                reply = Message(
                    to="negotiator@jabber.at",
                    body=msg.body,
                    metadata={"performative": "inform"}
                )
                await self.send(reply)

    async def setup(self):
        self.add_behaviour(self.RouteBehaviour())
