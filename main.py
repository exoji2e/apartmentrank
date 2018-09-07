import pathlib
import pickle
from collections import defaultdict
from dataclasses import dataclass

import requests
import feedparser
import geopy
from bs4 import BeautifulSoup


class Datastore:
    path = pathlib.Path('./datastore.pickle')
    data = defaultdict(lambda: dict())

    def __init__(self):
        if self.path.exists():
            with self.path.open("rb") as f:
                data = pickle.load(f)
                self.data.update(data)

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        self.save()

    def save(self):
        with open(self.path, "wb+") as f:
            pickle.dump(dict(self.data), f)


@dataclass
class Property:
    link: str
    address: str
    price: float
    sqm: float
    rooms: float
    monthly_fee: float

    def __repr__(self):
        return f"<{self.address}: {self.link}\n  Pris: {self.price}\n  Area: {self.sqm}\n  Fee: {self.monthly_fee}\n  Rooms: {self.rooms}\n  MoCost: {self.monthly_cost()}\n  MoCost/sqm: {self.monthly_cost_per_sqm()}>"

    def monthly_cost_per_sqm(self):
        return self.monthly_cost() / self.sqm

    def monthly_cost(self, downpayment=1_000_000, interest=0.02):
        return (((self.price - downpayment) * interest / 12) + self.monthly_fee)


db = Datastore()


def get_entry(link):
    if link not in db.data:
        r = requests.get(link)
        with db:
            db.data[link] = r.content
    return db.data[link]


def parse_page(title, link):
    soup = BeautifulSoup(get_entry(link), 'html.parser')

    priceprop = soup.find(class_="property__price")
    if priceprop:
        price = float("".join(priceprop.string.strip().split(" ")[:-1]))
        # print(price)
    else:
        print("Probably already sold")
        return

    monthly_fee = 0
    for p in soup.find(class_="property__attributes").find_all("dd"):
        txt = p.string
        if not txt:
            txt = p.contents[0]
        # &nbsp; to space
        txt = txt.replace("\xa0", " ")
        if "m²" == txt.split()[-1]:
            sqm = float(txt.split(" ")[0].replace(",", "."))
        elif "rum" in txt:
            rooms = float(txt.split(" ")[0].replace(",", "."))
        elif "kr/m" in txt and "n" in txt:
            num = "".join(txt.strip().split(" ")[:-1])
            monthly_fee = float(num.replace(",", "."))

    return Property(link, title, price, sqm, rooms, monthly_fee)


def get_coord(address):
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="apartmentbuyer")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    return None


    


def main():
    d = feedparser.parse('https://www.hemnet.se/mitt_hemnet/sparade_sokningar/15979794.xml')

    props = []
    for entry in d.entries:
        prop = parse_page(entry['title'], entry['link'])
        # coord = get_coord(entry['title'])
        # print(coord)

        if prop:
            props.append(prop)

    for prop in sorted(props, key=lambda p: p.monthly_cost_per_sqm()):
        print(prop)


if __name__ == "__main__":
    main()
