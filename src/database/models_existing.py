from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from src.database.base import Base

class Attraction(Base):
    __tablename__ = 'attractions'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))


class AttractionImage(Base):
    __tablename__ = 'attraction_images'
    id = Column(Integer, primary_key=True)
    attraction_id = Column(Integer, ForeignKey('attractions.id'))
    file_path = Column(Text, nullable=False)
    embedding = Column(Vector(128))