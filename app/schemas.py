from pydantic import BaseModel

class PredictionInput(BaseModel):
    developer: str
    complex_class: str
    district: str
    okrug: str
    ready_status: str
    metro_station: str
    metro_line: str
    deal_type: str
    total_area: float
    to_center_km: float
    mortgage_rate_at_listing: float
    completion_year: int
    years_to_completion: int
    year_posted: int
    month_posted: int

class PredictionOutput(BaseModel):
    predicted_price: float