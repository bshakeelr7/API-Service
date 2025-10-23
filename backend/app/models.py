from sqlalchemy import Column, Integer, String, Text
from .db import Base

class ModelMeta(Base):
    __tablename__ = "model_meta"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True, index=True, nullable=False)
    image_type = Column(String, index=True)
    minio_path = Column(String)
    framework = Column(String)   
    file_name = Column(String)
    description = Column(Text, nullable=True)
