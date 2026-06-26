from typing import Optional
from langchain.agents import AgentState


class TripState(AgentState):
    # Set by user input
    origin: str
    travelers: str
    # Set after Phase 1 (manager picks the cheapest combo)
    destination: Optional[str]
    trip_duration_nights: Optional[str]
    chosen_flight: Optional[str]
    chosen_hotel: Optional[str]
    total_cost: Optional[str]
