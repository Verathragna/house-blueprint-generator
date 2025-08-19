from typing import Dict, List, Optional
from pydantic import BaseModel, Field


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
    roomAdjacency: Optional[Dict[str, List[str]]] = None
    constraints: Optional[Dict[str, float]] = None
