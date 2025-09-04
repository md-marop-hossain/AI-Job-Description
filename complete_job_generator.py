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

    def _get_experience_level(self, years: float, job_title: str = "") -> str:
        prompt = f"""
You are an expert HR professional, career strategist, and workforce analyst 
with global expertise in professional hierarchies across industries. 
Your task is to classify the correct professional **experience level** 
based on years of experience and job title context.

### Input
Years of Experience: {years}
Job Title: "{job_title}"

---

### Analysis Framework

1. **Primary Criteria**
   - Use **years of experience** as a baseline indicator.
   - Adjust for **industry norms** and **career progression speed**:
     - Technology: accelerated progression; Senior possible after ~5–7 years.
     - Healthcare: longer timelines; Senior roles often require 10+ years & certifications.
     - Consulting: clear progression ladder (Analyst → Associate → Manager → Principal → Partner).
     - Academia: formal path (Assistant → Associate → Full Professor).
     - Corporate/Finance: managerial levels usually require more years.
     - Startups: titles may inflate, but still classify realistically.

2. **Job Title Context**
   - Titles with explicit seniority markers override years when appropriate:
     - "Intern", "Trainee", "Assistant", "Junior" → Early career levels regardless of years.
     - "Senior", "Lead", "Principal" → Elevated experience level.
     - "Manager", "Director", "Head of", "Vice President (VP)" → Executive ladder roles.
     - "Chief", "C-Level", "CEO/CTO/CFO/COO" → Top executive leadership.
   - Neutral roles like "Analyst" or "Specialist" depend more on years.

3. **Experience Level Classifications (Universal Set)**
   - **Entry Level** → Internships, trainees, < 1 year experience, early-career generalists.
   - **Junior** → 1–3 years, developing professional, growing autonomy.
   - **Mid-Level** → 3–7 years, competent, independent, some leadership/mentorship.
   - **Senior** → 7–12 years or explicit "Senior" title, advanced expertise, leads projects.
   - **Lead** → Technical/functional lead, expert authority, small team leadership.
   - **Principal** → Highly senior specialist or strategic leader, organization-wide impact.
   - **Manager/Director** → People management, operational or departmental leadership.
   - **Executive** → VP-level, Head of Function, broad strategic responsibility.
   - **C-Level** → Chief Officer titles, corporate-wide leadership, board-level strategy.

4. **Conflict Resolution (Years vs. Title)**
   - If years appear low but title suggests seniority:
     - Assume the **title context dominates** (e.g., "Senior Data Engineer" with 2 years should classify as "Senior").
   - If years are high but title is generic:
     - Default to a logically higher level (e.g., 15 years as just "Analyst" → at least "Senior").

---

### Output Instructions:
- Return **ONLY** the single most accurate career level term.
- Use exact classifications (no explanations): 
  "Entry Level", "Junior", "Mid-Level", "Senior", "Lead", "Principal", "Manager", "Director", "Executive", "C-Level".
- Do not output multiple options or add qualifiers like "Level".

### Examples
- 0.5 years, "Software Engineer" → Entry Level
- 2 years, "Junior Developer" → Junior
- 5 years, "Data Scientist" → Mid-Level
- 8 years, "Senior Analyst" → Senior
- 12 years, "Engineering Manager" → Manager
- 18 years, "Director of Finance" → Director
- 20 years, "Chief Technology Officer" → C-Level
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert HR professional and career strategist. Always return the single best professional experience level classification that matches the role and years of experience according to global standards."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=50
            )
            
            level = response.choices[0].message.content.strip()
            
            valid_levels = {
                "Entry Level", "Junior", "Mid-Level", "Senior",
                "Lead", "Principal", "Manager", "Director",
                "Executive", "C-Level"
            }
            
            if any(valid in level for valid in valid_levels):
                return level
            else:
                return self._fallback_experience_level(years)
                
        except Exception as e:
            print(f"Error inferring experience level with AI: {e}")
            return self._fallback_experience_level(years)

    def _fallback_experience_level(self, years: float, job_title: str = "") -> str:
        title_lower = job_title.lower()

        if any(keyword in title_lower for keyword in ["intern", "trainee", "apprentice"]):
            return "Entry Level"
        if any(keyword in title_lower for keyword in ["junior", "assistant", "associate"]):
            return "Junior"
        if any(keyword in title_lower for keyword in ["senior", "sr."]):
            return "Senior"
        if any(keyword in title_lower for keyword in ["lead", "head", "principal"]):
            return "Lead" if "principal" not in title_lower else "Principal"
        if any(keyword in title_lower for keyword in ["manager"]):
            return "Manager"
        if any(keyword in title_lower for keyword in ["director"]):
            return "Director"
        if any(keyword in title_lower for keyword in ["vp", "vice president"]):
            return "Executive"
        if any(keyword in title_lower for keyword in ["chief", "cfo", "ceo", "coo", "cto", "cio", "cmo", "c-level"]):
            return "C-Level"

        if years < 1:
            return "Entry Level"
        elif years < 3:
            return "Junior"
        elif years < 7:
            return "Mid-Level"
        elif years < 12:
            return "Senior"
        elif years < 18:
            return "Manager"
        elif years < 25:
            return "Director"
        else:
            return "Executive"

    def _infer_industry(self, job_title: str) -> str:
        prompt = f"""
You are an expert industry analyst with deep knowledge of global business sectors, 
emerging markets, and professional classification systems. Your task is to determine 
the most precise and professional industry classification for the given job title.

Job Title: "{job_title}"

### Analysis Guidelines:
1. **Contextual Understanding**
   - Carefully analyze the title for domain context (e.g., "Software Engineer" → Technology, not just Engineering in general).
   - Consider both primary function (core responsibilities implied) and domain relevance (the business or market where the job most likely exists).
   - Account for modern and emerging industries (e.g., FinTech, EdTech, HealthTech, E-commerce, AI/ML, Cybersecurity).

2. **Industry Identification**
   - Assign the job title to a clear, specific industry sector.
   - Use **recognizable and standardized industry terminology** as found in:
     - Professional job boards (LinkedIn, Indeed, Glassdoor)
     - Business and government classification systems (e.g., NAICS, ISIC, GICS, O*NET)
     - Industry reports and global workforce trends

3. **Specificity Requirement**
   - Be precise rather than generic:
     - Use "Financial Technology (FinTech)" instead of "Technology"
     - Use "Pharmaceuticals & Biotechnology" instead of just "Healthcare"
     - Use "E-commerce & Online Retail" instead of simply "Retail"
     - Use "Digital Media & Entertainment" instead of "Media"

4. **Dominant Industry**
   - If the role spans multiple industries, **select the dominant/primary industry** where the job most commonly exists.

5. **Output Rules**
   - Output ONLY the **industry name** (e.g., "Technology & Software Development" or "Financial Services & Banking")
   - Do NOT provide explanations, reasoning, or additional commentary.
   - Do NOT output multiple industries or options — just the single best classification.

### Examples of Good Classifications:
- "Technology & Software Development"
- "Financial Services & Banking"
- "Pharmaceuticals & Biotechnology"
- "E-commerce & Online Retail"
- "Digital Marketing & Advertising"
- "Energy & Renewable Utilities"
- "Transportation, Logistics & Supply Chain"
- "Consulting & Professional Services"
- "Aerospace & Defense"
- "Education & EdTech"
- "Legal & Compliance Services"
- "Manufacturing & Industrial Engineering"

Now return only the best fitting industry classification for the provided job title.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert AI trained to classify job titles into precise industries using globally recognized business and workforce classification standards. Always output the single most specific, professional industry name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            industry = response.choices[0].message.content.strip()
            
            if industry and len(industry) > 2 and len(industry) < 100:
                return industry
            else:
                return self._fallback_industry_inference(job_title)
                
        except Exception as e:
            print(f"Error inferring industry with AI: {e}")
            return self._fallback_industry_inference(job_title)

    def _fallback_industry_inference(self, job_title: str) -> str:
        title_lower = job_title.lower()

        industry_keywords = {
            "Technology & Software Development": [
                "software", "developer", "engineer", "programmer", "it", "coding",
                "devops", "architect", "qa", "web", "full stack", "backend", "frontend", "cloud"
            ],
            "Artificial Intelligence & Machine Learning": [
                "ai", "artificial intelligence", "ml", "machine learning", "nlp",
                "deep learning", "cv", "computer vision", "data scientist", "data science"
            ],
            "Cybersecurity & Information Security": [
                "cybersecurity", "infosec", "security analyst", "penetration tester", 
                "ethical hacker", "cryptography", "network security"
            ],
            "Data Analytics & Business Intelligence": [
                "data analyst", "data engineer", "bi", "business intelligence", "analytics"
            ],
            "Financial Services & Banking": [
                "finance", "bank", "investment", "accounting", "auditor",
                "analyst", "treasury", "audit", "equity", "loan", "credit", "wealth"
            ],
            "Financial Technology (FinTech)": [
                "fintech", "blockchain", "crypto", "defi", "digital payments",
                "mobile banking", "trading platform"
            ],
            "Healthcare & Medical Services": [
                "doctor", "nurse", "clinician", "hospital", "medical", "healthcare",
                "therapist", "physician", "dentist", "surgeon"
            ],
            "Pharmaceuticals & Biotechnology": [
                "biotech", "pharmaceutical", "drug", "clinical trials", "genomics", "molecular"
            ],
            "Education & Training": [
                "teacher", "professor", "lecturer", "instructor", "tutor",
                "education", "curriculum", "school", "learning", "academic"
            ],
            "Education Technology (EdTech)": [
                "edtech", "learning platform", "e-learning", "online courses",
                "education technology", "mooc"
            ],
            "Green Technology & Renewable Energy": [
                "renewable", "solar", "wind", "sustainability", "green energy",
                "climate", "environmental", "clean tech", "carbon", "energy transition"
            ],
            "Energy, Oil & Utilities": [
                "petroleum", "oil", "gas", "utilities", "hydropower", "nuclear"
            ],
            "Manufacturing & Industrial Engineering": [
                "production", "manufacturing", "assembly", "automation", "mechanical",
                "industrial", "plant", "maintenance", "technician", "fabricator"
            ],
            "Transportation, Logistics & Supply Chain": [
                "logistics", "supply chain", "transport", "delivery", "fleet",
                "shipping", "driver", "freight", "warehousing"
            ],
            "Retail, E-commerce & Consumer Goods": [
                "retail", "store", "sales", "associate", "merchandise", "e-commerce",
                "shopping", "customer support", "buyer", "shop manager"
            ],
            "Hospitality, Travel & Tourism": [
                "hotel", "hospitality", "restaurant", "chef", "catering",
                "travel", "tourism", "resort", "concierge", "bartender", "waiter"
            ],
            "Digital Marketing & Advertising": [
                "marketing", "seo", "sem", "social media", "advertising", "growth",
                "content", "campaign", "influencer", "branding", "performance marketing"
            ],
            "Creative, Design & Multimedia": [
                "designer", "graphic", "illustrator", "ux", "ui", "creative",
                "animator", "video editor", "photographer", "visual artist"
            ],
            "Media, Journalism & Entertainment": [
                "journalist", "reporter", "media", "broadcaster", "radio",
                "television", "producer", "actor", "film", "music", "entertainment"
            ],
            "Real Estate & Property": [
                "realtor", "broker", "real estate", "property", "leasing", "valuation"
            ],
            "Legal & Compliance Services": [
                "attorney", "lawyer", "legal", "solicitor", "paralegal",
                "judge", "compliance", "contracts", "regulatory"
            ],
            "Government, Policy & Public Administration": [
                "civil service", "government", "ambassador", "public sector", "policy",
                "municipal", "federal", "state", "regulatory officer"
            ],
            "Consulting & Professional Services": [
                "consultant", "strategy", "operations", "management",
                "executive", "advisor", "business consultant", "corporate strategy"
            ],
            "Non-Profit & Social Impact": [
                "non-profit", "charity", "ngo", "foundation", "volunteer",
                "community", "humanitarian", "social worker"
            ],
            "Aerospace & Defense": [
                "aerospace", "defense", "pilot", "aviation", "army", "airforce",
                "military", "naval", "space", "satellite"
            ],
            "Agriculture, Food & Farming": [
                "farmer", "agriculture", "horticulture", "crop", "food processing",
                "fisheries", "livestock"
            ]
        }

        for industry, keywords in industry_keywords.items():
            if any(kw in title_lower for kw in keywords):
                return industry

        tokens = set(title_lower.split())
        for industry, keywords in industry_keywords.items():
            if tokens.intersection(set(keywords)):
                return industry

        return "General Business & Services"

    def _build_prompt(self, params: JobParameters) -> str:
        exp_level = self._get_experience_level(params.experience, params.job_title)
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
