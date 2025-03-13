from sqlalchemy.orm import Session
from .models import Restaurant, Dish, Sale

def get_or_create_restaurant(session: Session, name: str):
    restaurant = session.query(Restaurant).filter_by(name=name).first()
    if not restaurant:
        restaurant = Restaurant(name=name)
        session.add(restaurant)
        session.commit()
    return restaurant

def get_or_create_dish(session: Session, name: str) -> Dish:
    dish = session.query(Dish).filter_by(name=name).first()
    if not dish:
        dish = Dish(name=name)
        session.add(dish)
        session.commit()
    return dish

def bulk_create_sales(session: Session, records: list[dict]):
    session.bulk_insert_mappings(Sale, records)
    session.commit()