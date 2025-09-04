import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class JobParams(BaseModel):
    job_title: str
    industry: str
    education: str
    company_name: str = "Your Company"
    location_type: str = "Hybrid"
    experience: float
    required_skills: List[str]
    preferred_skills: List[str]

class JobSections(BaseModel):
    Executive_Summary: str = Field(alias="Executive Summary")
    Key_Responsibilities: List[str] = Field(alias="Key Responsibilities")
    Required_Qualifications: List[str] = Field(alias="Required Qualifications")
    Preferred_Qualifications: List[str] = Field(alias="Preferred Qualifications")
    What_We_Offer: List[str] = Field(alias="What We Offer")
    skills: List[str]

    class Config:
        populate_by_name = True

class JobOutputs(BaseModel):
    sections: JobSections

class JobDescription(BaseModel):
    timestamp: str
    params: JobParams
    outputs: JobOutputs

@dataclass
class JobParameters:
    job_title: str
    experience: float
    education: str
    location_type: Optional[str] = None
    required_skills: Optional[List[str]] = None

class AIJobDescriptionGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass it as a parameter."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _build_prompt(self, params: JobParameters) -> str:
        location = params.location_type or "INFER and choose the best location type (Remote, Hybrid, On-site)"
        required = json.dumps(params.required_skills) if params.required_skills else "INFER and generate relevant required skills list"
        
        prompt = f"""
You are an expert AI assistant specialized in creating detailed, professional job descriptions.

Generate a complete JSON job description matching the exact format below. Return only the JSON.

Input:
- Job Title: {params.job_title}
- Experience: {params.experience} years
- Education: {params.education}
- Location Type: {location}
- Required Skills: {required}

Determine the precise industry classification for the job title (e.g., "Financial Technology (FinTech)" instead of "Technology").

Return only one industry name.

Required JSON Structure:
{{
  "timestamp": "{datetime.now().isoformat()}",
  "params": {{
    "job_title": "{params.job_title}",
    "industry": "INFER from job title",
    "Education": "{params.education}",
    "company_name": "Your Company",
    "location_type": "{location}",
    "experience": {params.experience},
    "required_skills": {required},
    "preferred_skills": [
      "Generate 3-4 complementary skills for {params.job_title}"
    ]
  }},
  "outputs": {{
    "sections": {{
      "Executive Summary": "Provide a compelling overview of the {params.job_title} opportunity at Your Company, emphasizing workplace culture, strategic impact, and the need for {params.experience} years experience.",
      "Key Responsibilities": [
        "Generate 5-7 responsibilities for a {params.job_title} role, crafted for someone with {params.experience} years of relevant experience. Use strong action verbs, specifying accountability, leadership, collaboration, technical/functional, and growth-focused duties."
      ],
      "Required Qualifications": [
        "List 4-6 non-negotiable requirements specifically tailored for {params.job_title}, including minimum education ({params.education}), {params.experience} years of experience, certifications, and essential hard/soft skills."
      ],
      "Preferred Qualifications": [
        "List 3-4 additional attributes that would strengthen a candidate for this {params.job_title} (examples: advanced technologies, industry awards, specializations, leadership, bilingual ability, etc)."
      ],
      "What We Offer": [
        "Describe 4-5 attractive benefits including growth, career development, work-life balance, competitive compensation, health/wellness, and workplace flexibility."
      ],
      "skills": [
        "List every unique tool, programming language, technology, and soft skill explicitly mentioned in the job description sections (e.g., SQL, Excel, Python, Tableau, Power BI, communication, analytical thinking), with no duplicates."
      ]
    }}
  }}
}}

Instructions:
- job_title, education, and experience are always required and must be explicitly referenced throughout the output.
- If location_type or required_skills are missing, infer and generate them appropriately for the role, education, and experience context.
- Make all sections rich, detailed, authentic, and professional—no placeholders.
- Tailor all content to the provided job_title, experience, education, and inferred parameters.
- Return ONLY valid JSON matching this exact structure, fully populated with production-ready content.
"""
        return prompt

    def generate(self, params: JobParameters) -> str:
        prompt = self._build_prompt(params)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert job description generator. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content

def save_job_description(job_desc: Dict[str, Any], output_dir: str = "output") -> None:
    os.makedirs(output_dir, exist_ok=True)
    title = job_desc["params"]["job_title"]
    clean = "".join(c for c in title if c.isalnum() or c in (" ", "_")).replace(" ", "_").lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{clean}_{ts}.json"
    path = Path(output_dir) / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job_desc, f, indent=2, ensure_ascii=False)

def load_config(config_path: str = "config.json") -> JobParameters:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return JobParameters(
        job_title=cfg["job_title"],
        experience=cfg["experience"],
        education=cfg["education"],
        location_type=cfg.get("location_type"),
        required_skills=cfg.get("required_skills")
    )

def main():
    params = load_config()
    generator = AIJobDescriptionGenerator()
    job_desc_json = generator.generate(params)
    job_desc = json.loads(job_desc_json)
    save_job_description(job_desc)

if __name__ == "__main__":
    main()
