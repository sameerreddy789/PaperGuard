from pydantic import BaseModel, Field
from typing import Optional, List

class Reference(BaseModel):
    """
    Pydantic data model representing a parsed academic reference.
    """
    authors: List[str] = Field(default_factory=list, description="List of author names")
    title: str = Field(..., description="Title of the paper or book")
    year: Optional[int] = Field(None, description="Year of publication")
    journal: Optional[str] = Field(None, description="Journal or conference name")
    doi: Optional[str] = Field(None, description="Digital Object Identifier (DOI)")
    volume: Optional[str] = Field(None, description="Volume of the journal")
    pages: Optional[str] = Field(None, description="Page numbers")
    raw_text: str = Field(..., description="The original raw reference text")
