import logging
import pathlib
import pickle
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict
from dataclasses import dataclass

import requests
import feedparser
from geopy import distance
from bs4 import BeautifulSoup
from tabulate import tabulate


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("geopy").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
log = logging.getLogger(__name__)


class Datastore:
    path = pathlib.Path('./datastore.pickle')
    data: Dict[str, Dict[str, Any]] = defaultdict(lambda: dict())

    def __init__(self) -> None:
        if self.path.exists():
            with self.path.open("rb") as f:
                data = pickle.load(f)
                self.data.update(data)

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        self.save()

    def save(self) -> None:
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
    coords: Optional[Tuple[float, float]] = None

    def __repr__(self):
        return f"<{self.address}: {self.link}\n  Pris: {self.price}\n  Area: {self.sqm}\n  Fee: {self.monthly_fee}\n  Rooms: {self.rooms}\n  MoCost: {self.monthly_cost()}\n  MoCost/sqm: {self.monthly_cost_per_sqm()}\n  Dist to Lund C: {self.distance_to(lund_c_coords)}>"

    @staticmethod
    def headers() -> List[str]:
        return ["address", "price", "area", "fee", "rooms", "cost/mo", "cost/sqm/mo", "Dist to Lund C"]

    def row(self) -> List[Any]:
        return [
            self.address,
            #self.link,
            self.price,
            self.sqm,
            self.monthly_fee,
            self.rooms,
            self.monthly_cost(),
            self.monthly_cost_per_sqm(),
            self.distance_to(lund_c_coords)
        ]

    def monthly_cost_per_sqm(self) -> float:
        return self.monthly_cost() / self.sqm

    def monthly_cost(self, downpayment=800_000, interest=0.016, taxreduction=0.3) -> float:
        return ((self.price - downpayment) * (interest * (1 - taxreduction) / 12)) + self.monthly_fee

    def distance_to(self, other: Tuple[float, float]) -> Optional[float]:
        if not self.coords:
            return None
        return distance.distance(self.coords, other).km


db = Datastore()


def get_entry(link: str) -> str:
    if "page" not in db.data[link]:
        r = requests.get(link)
        with db:
            db.data[link].update(page=r.content)
    else:
        print(db.data)
        log.debug(f"Got {link} from cache")
    return db.data[link]["page"]


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


def get_coord(address) -> Optional[Tuple[float, float]]:
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="apartmentbuyer")
    location = geolocator.geocode(address + ", Lund")
    if location:
        return (location.latitude, location.longitude)
    return None


lund_c_coords = (55.7068, 13.187)


def crawl() -> List[Property]:
    print("Crawling...")
    d = feedparser.parse('https://www.hemnet.se/mitt_hemnet/sparade_sokningar/15979794.xml')

    for entry in d.entries:
        title, link = entry['title'], entry['link']
        if not db.data[link]:
            prop = parse_page(title, link)
            if prop:
                db.data[link]["property"] = prop

    assign_coords()
    db.save()
    return [v["property"] for v in db.data.values() if "property" in v]


def assign_coords() -> None:
    print("Mapping addresses to coordinates...")
    props = [v["property"] for v in db.data.values() if "property" in v]
    for prop in props:
        if not prop.coords:
            prop.coords = get_coord(prop.address)


def main() -> None:
    props = crawl()

    tableprops = [Property.headers()] + [
        p.row()
        for p in sorted(props, key=lambda p: p.monthly_cost_per_sqm())
    ]
    print(tabulate(tableprops[1:20], headers=tableprops[0]))


if __name__ == "__main__":
    main()
