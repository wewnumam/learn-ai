import asyncio
from agents.citizen import CitizenAgent
from agents.router import RouterAgent
from agents.negotiator import NegotiatorAgent
from agents.department import DepartmentAgent
from agents.supervisor import SupervisorAgent

async def main():
    agents = [
        CitizenAgent("citizen@jabber.at", "123"),
        RouterAgent("router@jabber.at", "123"),
        NegotiatorAgent("negotiator@jabber.at", "123"),
        DepartmentAgent("department@jabber.at", "123"),
        SupervisorAgent("supervisor@jabber.at", "123"),
    ]

    for a in agents:
        await a.start()

    await asyncio.sleep(30)

    for a in agents:
        await a.stop()

asyncio.run(main())