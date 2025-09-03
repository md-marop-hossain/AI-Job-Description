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
    experience_level: str
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

    def _get_experience_level(self, years: float) -> str:
        if years <= 2:
            return "Entry"
        elif years <= 7:
            return "Mid"
        else:
            return "Senior"

    def _infer_industry(self, job_title: str) -> str:
        title_lower = job_title.lower()
        industry_keywords = {
            "Technology": ["software", "developer", "engineer", "it", "ai", "ml", "data", "devops", "architect", "cyber", "cloud", "qa", "web", "systems"],
            "Marketing & Advertising": ["marketing", "brand", "campaign", "content", "digital", "seo", "sem", "social media", "advertising", "copywriter"],
            "Financial Services": ["finance", "accounting", "analyst", "banking", "investment", "treasury", "audit"],
            "Healthcare": ["doctor", "nurse", "medical", "health", "clinical", "pharma", "hospital", "physician", "dentist", "therapist"],
            "Education": ["teacher", "professor", "education", "instructor", "curriculum", "lecturer", "tutor"],
            "Business": ["analyst", "manager", "consultant", "operations", "business", "executive", "administrator"],
            "Creative & Design": ["designer", "creative", "artist", "illustrator", "ux", "ui", "graphic", "videographer", "photographer"],
            "Government & Public Service": ["civil", "public", "policy", "government", "ambassador", "officer", "regulatory"],
            "Trades & Manufacturing": ["technician", "mechanic", "electrician", "welder", "craftsman", "plumber", "manufacturing", "production"],
            "Legal": ["lawyer", "attorney", "legal", "paralegal", "solicitor", "judge", "compliance"],
            "Retail & Hospitality": ["sales", "associate", "store", "retail", "hospitality", "waiter", "chef", "hotel", "restaurant"]
        }
        for industry, keywords in industry_keywords.items():
            if any(kw in title_lower for kw in keywords):
                return industry
        return "General"

    def _generate_default_skills(self, job_title: str, experience_level: str) -> List[str]:
        entry_skills = [
            "Communication", "Problem Solving", "Learning Agility", "Adaptability", "Teamwork", "Reliability"
        ]
        mid_skills = [
            "Project Management", "Collaboration", "Critical Thinking", "Organization", "Detail Orientation", "Time Management", "Analytical Thinking"
        ]
        senior_skills = [
            "Strategic Leadership", "Mentoring", "Innovation", "Decision Making", "Negotiation", "Vision Setting"
        ]
        title_lower = job_title.lower()
        tech_skills = ["Programming", "Version Control", "Software Development", "Testing", "Code Review"]
        business_skills = ["Business Analysis", "Financial Literacy", "Operations Management", "Research", "Reporting"]
        marketing_skills = ["Digital Marketing", "Content Creation", "SEO", "Brand Management"]
        healthcare_skills = ["Patient Care", "Clinical Procedures", "Medical Documentation"]
        cluster_skills = []
        if any(kw in title_lower for kw in ["developer", "engineer", "software", "it", "programmer"]):
            cluster_skills = tech_skills
        elif any(kw in title_lower for kw in ["analyst", "manager", "business", "consultant", "operations", "executive"]):
            cluster_skills = business_skills
        elif any(kw in title_lower for kw in ["marketing", "brand", "seo", "digital"]):
            cluster_skills = marketing_skills
        elif any(kw in title_lower for kw in ["doctor", "nurse", "medical", "health", "clinical"]):
            cluster_skills = healthcare_skills
        if experience_level == "Entry":
            skills = entry_skills + cluster_skills[:3]
        elif experience_level == "Mid":
            skills = mid_skills + cluster_skills[:5]
        else:
            skills = senior_skills + cluster_skills
        seen = set()
        universal_skills = [s for s in skills if not (s in seen or seen.add(s))]
        return universal_skills

    def _build_prompt(self, params: JobParameters) -> str:
        exp_level = self._get_experience_level(params.experience)
        industry = self._infer_industry(params.job_title)
        location = params.location_type
        required = params.required_skills
        timestamp = datetime.now().isoformat()
        prompt = f"""
You are an expert AI assistant specializing in creating fully detailed, professional job descriptions for ANY role, industry, or career background.

Generate a complete, publish-ready job description in JSON format, with EXACTLY the following structure. Return ONLY the JSON (no other text).

Input Parameters:
- Job Title: {params.job_title}
- Experience Required: {params.experience} years
- Education: {params.education}
- Location Type: {location if location else "INFER and choose the best location type based on the role, e.g., Remote, Hybrid, On-site."}
- Required Skills: {required if required else "INFER and generate a professional, relevant skill set for the role, education, and experience level."}

Required JSON Structure:
{{
  "timestamp": "{timestamp}",
  "params": {{
    "job_title": "{params.job_title}",
    "industry": "{industry}",
    "Education": "{params.education}",
    "experience_level": "{exp_level}",
    "company_name": "Your Company",
    "location_type": "{location if location else "INFER AND SET"}",
    "experience": {params.experience},
    "required_skills": {json.dumps(required) if required else "INFER AND SET AS LIST"},
    "preferred_skills": [
      "Generate 3-4 realistic, complementary skills for {params.job_title} at {exp_level} level"
    ]
  }},
  "outputs": {{
    "sections": {{
      "Executive Summary": "Provide a compelling overview of the {params.job_title} ({exp_level} level) opportunity at Your Company, emphasizing workplace culture and strategic impact, and referencing the need for {params.experience} years experience.",
      "Key Responsibilities": [
        "Generate 5-7 responsibilities for a {params.job_title} at {exp_level} level, each crafted for someone with {params.experience} years of relevant experience. Use strong action verbs and be as specific as possible. Include accountability, leadership, collaboration, technical/functional, and growth-focused duties."
      ],
      "Required Qualifications": [
        "List 4-6 non-negotiable requirements specifically tailored for {params.job_title}, including minimum education ({params.education}), {params.experience} years of experience, certifications, and essential hard/soft skills. Be precise and descriptive."
      ],
      "Preferred Qualifications": [
        "List 3-4 additional attributes that would particularly strengthen a candidate for this {params.job_title} at {exp_level} level (examples: advanced technologies, industry awards, specializations, leadership, bilingual ability, etc)."
      ],
      "What We Offer": [
        "Describe 4-5 attractive benefits for a {params.job_title} ({exp_level}), focusing on growth, career development, work-life balance, competitive compensation, health/wellness, and workplace flexibility."
      ],
      "skills": [
        "List every unique tool, programming language, technology, and soft skill explicitly mentioned in Key Responsibilities, Required Qualifications, Preferred Qualifications, and any part of the job description (including e.g., SQL, Excel, Python, Tableau, Power BI, communication, analytical thinking, etc.). Ensure all such items are included as separate skills, with no duplicates."
      ]
    }}
  }}
}}

Instructions:
- job_title, education, and experience are always required and must be explicitly referenced throughout the output.
- If location_type or required_skills are not provided, you must infer and generate them appropriately, choosing the most suitable values for {params.job_title} and education/experience context. Use professional and current industry-standard defaults.
- Make every section rich, detailed, authentic, and professional—no placeholders.
- Tailor all content to the provided job_title, experience, education, and inferred parameters.
- All content must be role-specific, meaningful, and meet current industry standards.
- Return ONLY valid JSON matching this exact structure. All arrays/fields must be fully populated with production-ready content.
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
