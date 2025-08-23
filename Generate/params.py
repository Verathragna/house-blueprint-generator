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
        for room, adjacent in value.items():
            if not isinstance(adjacent, list) or not all(
                isinstance(a, str) for a in adjacent
            ):
                raise ValueError("Adjacency mapping must be a list of strings")
            if len(adjacent) == 0:
                raise ValueError(f"Adjacency list for '{room}' cannot be empty")
        return value


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

