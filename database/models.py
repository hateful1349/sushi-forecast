from sqlalchemy import Column, Integer, String, Date, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Restaurant(Base):
    __tablename__ = 'restaurants'
    id = Column(Integer, primary_key=True)
    original_id = Column(Integer, unique=True)
    name = Column(String(150))
    sales = relationship("Sale", back_populates="restaurant")


class Dish(Base):
    __tablename__ = 'dishes'
    id = Column(Integer, primary_key=True)
    name = Column(String(300), unique=True)
    sales = relationship("Sale", back_populates="dish")


class Sale(Base):
    __tablename__ = 'sales'
    __table_args__ = (
        Index('ix_restaurant_date', 'restaurant_id', 'date'),
        Index('ix_dish_date', 'dish_id', 'date')
    )
    id = Column(Integer, primary_key=True)
    date = Column(Date, index=True)
    amount = Column(Integer)
    restaurant_id = Column(Integer, ForeignKey('restaurants.id'), index=True)
    dish_id = Column(Integer, ForeignKey('dishes.id'), index=True)

    restaurant_id = Column(Integer, ForeignKey('restaurants.id'), index=True)
    dish_id = Column(Integer, ForeignKey('dishes.id'), index=True)

    restaurant = relationship("Restaurant", back_populates="sales")
    dish = relationship("Dish", back_populates="sales")


class Forecast(Base):
    __tablename__ = 'forecasts'
    id = Column(Integer, primary_key=True)
    date = Column(Date)
    dish_name = Column(String(300))
    amount = Column(Integer)
    restaurant_id = Column(Integer)