"""Pydantic models for resume/CV extraction response."""

from typing import Optional

from pydantic import BaseModel, Field


class ExperienceItem(BaseModel):
    """A single work experience entry."""

    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company or organization name")
    dates: Optional[str] = Field(None, description="Employment period (e.g. 2020 - Present)")
    description: Optional[str] = Field(None, description="Brief description of role or achievements")


class EducationItem(BaseModel):
    """A single education entry."""

    degree: str = Field(..., description="Degree or qualification")
    institution: str = Field(..., description="School or university name")
    dates: Optional[str] = Field(None, description="Attendance period")


class ResumeExtraction(BaseModel):
    """Structured resume/CV data extracted from a document."""

    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Primary phone number")
    phones: list[str] = Field(
        default_factory=list,
        description="All detected phone numbers (deduplicated)",
    )
    summary: Optional[str] = Field(None, description="Professional summary or objective")
    experience: list[ExperienceItem] = Field(
        default_factory=list,
        description="Work experience entries",
    )
    education: list[EducationItem] = Field(
        default_factory=list,
        description="Education entries",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="List of skills or competencies",
    )
