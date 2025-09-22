from typing import Dict, List, Optional
from pydantic import BaseModel, Field, RootModel, field_validator


class Bathrooms(BaseModel):
    full: int = Field(default=2, ge=0)
    half: int = Field(default=0, ge=0)


class Garage(BaseModel):
    attached: bool = True
    carCount: Optional[int] = Field(default=None, ge=0)
    doorSizes: Optional[List[str]] = None


class Dimensions(BaseModel):
    width: float = Field(gt=0)
    depth: float = Field(gt=0)


class Adjacency(RootModel[Dict[str, List[str]]]):
    @field_validator("root")
    @classmethod
    def _check_lists(cls, value: Dict[str, List[str]]) -> Dict[str, List[str]]:
        case_map: Dict[str, str] = {}
        adjacency_map: Dict[str, set[str]] = {}
        for room, adjacent in value.items():
            if not isinstance(adjacent, list) or not all(isinstance(a, str) for a in adjacent):
                raise ValueError("Adjacency mapping must be a list of strings")
            room_label = room.strip()
            if not room_label:
                raise ValueError("Adjacency keys cannot be empty")
            room_key = room_label.lower()
            case_map.setdefault(room_key, room_label)
            adjacency_map.setdefault(room_key, set())
            seen: set[str] = set()
            for entry in adjacent:
                neighbor = entry.strip()
                if not neighbor:
                    raise ValueError(f"Adjacency entry for '{room_label}' cannot be empty")
                neighbor_key = neighbor.lower()
                if neighbor_key == room_key:
                    raise ValueError(f"Adjacency for '{room_label}' cannot reference itself")
                if neighbor_key in seen:
                    continue
                seen.add(neighbor_key)
                case_map.setdefault(neighbor_key, neighbor)
                adjacency_map[room_key].add(neighbor_key)
        for room_key, neighbors in list(adjacency_map.items()):
            for neighbor_key in neighbors:
                adjacency_map.setdefault(neighbor_key, set()).add(room_key)
        normalized: Dict[str, List[str]] = {}
        for room_key, neighbors in adjacency_map.items():
            room_label = case_map[room_key]
            normalized[room_label] = sorted(case_map[nk] for nk in neighbors)
        return normalized



class Constraints(RootModel[Dict[str, float]]):
    @field_validator("root")
    @classmethod
    def _positive(cls, value: Dict[str, float]) -> Dict[str, float]:
        for k, v in value.items():
            if v <= 0:
                raise ValueError("Constraint values must be positive")
        return value


class Params(BaseModel):
    houseStyle: Optional[str] = None
    dimensions: Optional[Dimensions] = None
    foundationType: Optional[str] = None
    stories: Optional[int] = Field(default=1, ge=1)
    bedrooms: int = Field(default=3, ge=0)
    bathrooms: Bathrooms = Bathrooms()
    kitchen: int = Field(default=1, ge=1)
    livingRooms: int = Field(default=1, ge=1)
    diningRooms: int = Field(default=1, ge=1)
    laundryRooms: int = Field(default=1, ge=1)
    bonusRoom: Optional[bool] = False
    garage: Optional[Garage] = None
    fireplace: Optional[bool] = False
    ownerSuiteLocation: Optional[str] = None
    masterBathOption: Optional[str] = None
    ceilingHeight: Optional[float] = Field(default=None, gt=0)
    vaultedCeilings: Optional[Dict[str, bool]] = None
    windowHeight: Optional[float] = Field(default=None, gt=0)
    doorHeight: Optional[float] = Field(default=None, gt=0)
    ada: Optional[bool] = None
    adaFeatures: Optional[Dict[str, bool]] = None
    attic: Optional[bool] = False
    adjacency: Optional[Adjacency] = None
    constraints: Optional[Constraints] = None

