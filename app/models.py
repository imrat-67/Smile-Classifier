from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from app.database import Base

class ClassificationHistory(Base):
    __tablename__ = "classification_history"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)