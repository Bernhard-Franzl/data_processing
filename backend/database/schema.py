from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

class Base(DeclarativeBase):
    pass

class Event(Base):
    __tablename__ = "event"
    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("room.id"), nullable=False)
    time: Mapped[datetime] = mapped_column(nullable=False)
    event_type: Mapped[int] = mapped_column(nullable=False)
    in_support_count: Mapped[int] = mapped_column(nullable=False)
    out_support_count: Mapped[int] = mapped_column(nullable=False)
    sensor_one_support_count: Mapped[int] = mapped_column(nullable=False)
    sensor_two_support_count: Mapped[int] = mapped_column(nullable=False)
    
    def __repr__(self):
        return f"<Event {self.id}>"
    
class Door(Base):
    __tablename__ = "door"
    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(nullable=False)
    
    def __repr__(self):
        return f"<Door {self.id}>"
