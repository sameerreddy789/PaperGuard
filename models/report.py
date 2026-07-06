from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .reference import Reference

class AgentResult(BaseModel):
    """
    Results from a specific verification agent.
    """
    agent_name: str = Field(..., description="Name of the agent (e.g., ReferenceAgent)")
    status: str = Field(..., description="Status of the check (e.g., passed, failed, warning)")
    findings: List[str] = Field(default_factory=list, description="List of specific findings or issues")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context or data from the agent")

class Report(BaseModel):
    """
    Overall verification report for a given paper.
    """
    paper_title: Optional[str] = Field(None, description="Extracted title of the paper")
    file_name: str = Field(..., description="Name of the parsed PDF file")
    summary: str = Field(..., description="High-level summary of the verification results")
    agent_results: List[AgentResult] = Field(default_factory=list, description="Detailed results from all agents")
    extracted_references: List[Reference] = Field(default_factory=list, description="References found in the paper")
