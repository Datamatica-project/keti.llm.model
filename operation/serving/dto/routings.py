from typing import TypedDict, Literal

RouteType = Literal["general_chat", "document_search"]

class RoutingResult(TypedDict):
    route: RouteType
    reasoning: str